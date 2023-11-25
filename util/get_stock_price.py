import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, time, timedelta, date
import holidays

def is_us_holiday(date_obj):
    us_holidays = holidays.US()
    return date_obj in us_holidays

def get_next_working_day(date_obj):
    while True:
        date_obj += timedelta(days=1)
        if date_obj.weekday() < 5 and not is_us_holiday(date_obj):
            return date_obj

def get_open_close_prices(symbol, timestamp):
    try:
        #print("symbol: ", symbol)
        #print("timestamp: ", timestamp)

        date_obj = datetime.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y")
        #print(date_obj)
        start_date = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
        end_date = datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, 0)

        market_closing_time = time(16, 0, 0)

        if date_obj.time() <= market_closing_time:
            if is_us_holiday(date_obj.date()):
                #print("Tweet made on a public holiday")
                date_obj = get_next_working_day(date_obj.date())
                #print("Updated date object for the next working date: ", date_obj)

            # Check if the date falls on a weekend (Saturday or Sunday) or after Friday market closing
            if date_obj.weekday() == 5:  # Saturday
                #print("Tweet made on Saturday")
                date_obj += timedelta(days=2)
                #print("Updating date_obj to Monday: ", date_obj)
            elif date_obj.weekday() == 6:  # Sunday
                #print("Tweet made on Saturday")
                date_obj += timedelta(days=1)
                #print("Updating date_obj to Monday: ", date_obj)

            start_date = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
            end_date = datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, 0)
            #print("start_date same day: ", start_date)
            #print("end_date same day: ", end_date)

        elif date_obj.time() > market_closing_time:
            if is_us_holiday(date_obj.date()):
                #print("Tweet on Public Holiday, Updating to fetch price on Working day")
                date_obj = get_next_working_day(date_obj.date())
                start_date = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
                end_date = datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, 0)
                #print("start_date next day: ", start_date)
                #print("end_date next day: ", end_date)
            else:
                # Check if the date falls on a weekend (Saturday or Sunday) or after Friday market closing
                if date_obj.weekday() == 5:
                    #print("Tweet made on Saturday")
                    date_obj += timedelta(days=2)
                    #print("Updating date_obj to Monday: ", date_obj)
                elif date_obj.weekday() == 6:
                    #print("Tweet made on Sunday")
                    date_obj += timedelta(days=1)
                    #print("Updating date_obj to Monday: ", date_obj)
                elif date_obj.weekday() == 4 and date_obj.time() > market_closing_time:  # After Friday market closing
                    #print("Tweet made on Friday, After market closing")
                    date_obj += timedelta(days=3)
                    #print("Updating date_obj to Monday: ", date_obj)
                else:
                    #print("Tweet made after market closing on ", date_obj)
                    date_obj += timedelta(days=1)
                    #print("Updating date_obj to next day: ", date_obj)

                start_date = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
                end_date = datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, 0)
                #print("start_date next day: ", start_date)
                #print("end_date next day: ", end_date)

        data = yf.download(symbol, start=start_date, end=end_date)
        #print(data)
        if data.empty:
            open_price, close_price = None, None
        else:
            open_price = data['Open'].iloc[0]
            close_price = data['Close'].iloc[0]

    except Exception as e:
        open_price, close_price = None, None

    return open_price, close_price


