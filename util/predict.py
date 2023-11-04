from __future__ import absolute_import, division, print_function
import os
from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
import csv
import nltk
import numpy as np
import logging
from transformers import AutoTokenizer


model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree

class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.

    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'

    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                          len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding


        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          agree=agree))
    return features


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]

def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def predict(text, model, write_to_csv=False, path=None, use_gpu=False, gpu_name='cuda:0', batch_size=5):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.
    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    use_gpu: (optional): bool 
        enables inference on GPU
    gpu_name: (optional): string
        multi-gpu support: allows specifying which gpu to use
    batch_size: (optional): int
        size of batching chunks
    """
    model.eval()
    nltk.download('punkt')

    sentences = [text] #sent_tokenize(text)
    #print('debug',type(sentences))
    device = gpu_name if use_gpu and torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s " % device)
    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logitn1', 'logitn2sfmx','prediction', 'sentiment_score'])
    for batch in chunks(sentences, len(sentences)):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        with torch.no_grad():
            model     = model.to(device)

            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            n1 = np.array([logits[0]])
            #print('debug',n1)
            logging.info(logits)
            logits = softmax(np.array(logits.cpu()))
            n2 = np.array([logits[0]])
            print('debug',n2)
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {'sentence': batch,
                            'logitn1': list(n1),
                            'logitn2sfmx': list(n2),
                            'prediction': predictions,
                            'sentiment_score': sentiment_score}

            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path, sep=',', index=False)

    return result

def predict_for_csv(tweets_path,output_path):
    """
    Take a .csv or .txt file as input, with path $tweets_path, output the prediction as finBERT_predictions.csv to path $output_path.
    In finBERT_predictions.csv, there will be ['sentence', 'logitn1', 'logitn2sfmx','prediction', 'sentiment_score'] as header.
    sentence: 
        corresponding sentence of prediction
    logitn1:
        embedding before softmax function
    logitn2sfmx:
        embedding after softmax function (probability of ['positive', 'negative', 'neutral'])
    prediction:
        the sentiment prediction with ['positive', 'negative', 'neutral']
    sentiment_score:
        defined as probability of positive substrat probability of negative, with range [-1,1]
    """
    # Required directory of tweets_data path and prediction output_data path
    output_file = 'finBERT_predictions.csv'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Read the tweets from .csv/.txt file to data_list, DONE：NO need to remove punctuations
    with open(tweets_path,'r') as f:
        reader = []
        readed_tweets = csv.reader(f)
        for row in readed_tweets:
            reader.append(row)

    data_list = [] 
    for i in range(len(reader)):
        data_list.append(reader[i][0])

    # Get prediction of data_list
    result_list =pd.DataFrame(columns=['sentence', 'logitn1', 'logitn2sfmx','prediction', 'sentiment_score'])
    for tw in data_list:
        result = predict(tw,model,write_to_csv=False) 
        result_list = pd.concat([result_list, result], ignore_index=True)

    result_list.to_csv(os.path.join(output_path,output_file), index=False)

def predict_for_singletext(text):
    """
    This funtion take a sentence(string) as a input, and return a list contains logitsn1, logitsn2sfmx, 
    predictions, score of this single sentence.

    Note that this funtion will throw away the original sentence in output.

    result[0]:
        logitsn1
    result[1]:
        logitsn2sfmx
    result[2]:  
        predictions
    result[3]:
        sentiment_score
    """
    predicted = predict(text,model,write_to_csv=False) 
    for index, row in predicted.iterrows():
        result = row.tolist()
        result.pop(0)
    return result