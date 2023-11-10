from predict import *

'''
# Test case for .csv/.txt input
input_path = './dataset/input_tweets.csv'
output_path = './dataset'
predict_for_csv(input_path,output_path)
'''

#Test case for sinlge text input
text = 'VIDEO: “I was in my office. I was minding my own business...” –David Solomon tells $GS interns how he learned he wa… https://t.co/QClAITywXV'
pred = predict_for_singletext(text)
#Use index to get data
for i in range(len(pred)):
    print(pred[i])
