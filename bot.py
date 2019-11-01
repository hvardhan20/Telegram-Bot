"""
Author: Harshavardhan
"""
import telepot
import os
import webbrowser as wb

# for others
# token  = input("Enter your token number")
token = 'YOUR_BOT_TOKEN_HERE'
TelegramBot = telepot.Bot(token)

import json
import requests
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import telepot
import keras


def telegram_bot_sendtext(bot_message):
    bot_token = 'YOUR_BOT_TOKEN_HERE'
    bot_chatID = 'YOUR_CHAT_ID_HERE'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


URL = "https://api.telegram.org/bot{}/".format(token)

text_1 = 'a'

while True:
    def get_url(url):
        response = requests.get(url)
        content = response.content.decode("utf8")
        return content


    def get_json_from_url(url):
        content = get_url(url)
        js = json.loads(content)
        return js


    def get_updates():
        url = URL + "getUpdates"
        js = get_json_from_url(url)
        return js


    def get_last_chat_id_and_text(updates):
        num_updates = len(updates["result"])
        last_update = num_updates - 1
        text = updates["result"][last_update]["message"]["text"]
        chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        return (text, chat_id)


    def send_message(text, chat_id):
        url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
        get_url(url)


    text, chat = get_last_chat_id_and_text(get_updates())
    # send_message(text, chat)
    # text1 = 'hi'
    import ctypes

    # print(text)

    if text == 'Lock' or text == 'lock':
        ctypes.windll.user32.LockWorkStation()

    elif text.lower() == 'facebook' and text_1 != text.lower():
        wb.open('www.facebook.com')
        text_1 = 'facebook'
        test = telegram_bot_sendtext("Opened Facebook")
        # print(test)

    elif text.lower() == 'google' and text_1 != text.lower():
        wb.open('www.google.com')
        test = telegram_bot_sendtext("Opened Google")
        text_1 = 'google'


    elif text.lower() == 'youtube' and text_1 != text.lower():
        wb.open('www.youtube.com')
        test = telegram_bot_sendtext("Opened Youtube")
        text_1 = 'youtube'

    elif text.lower() == 'github' and text_1 != text.lower():
        wb.open('www.github.com')
        text_1 = 'github'

    elif text.lower() == 'quora' and text_1 != text.lower():
        wb.open('www.quora.com')
        text_1 = 'quora'

    elif text.lower() == 'linkedin' and text_1 != text.lower():
        wb.open('www.linkedin.com')
        text_1 = 'linkedin'


    elif text.lower() == 'elearning' and text_1 != text.lower():
        wb.open('https://elearning.utdallas.edu/webapps/portal/execute/tabs/tabAction?tab_tab_group_id=_1_1')
        text_1 = 'elearning'



    elif text.lower() == 'stop':
        break


    elif text.lower() == 'project' and text_1 != text.lower():
        text_1 = 'project'
        # Using age and class to predicted survived

        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # read the train and test dataset
        train_data = pd.read_csv('C:\\Users\\Shyam\\Desktop\\pythonprojects\\datasets\\train-data.csv')
        test_data = pd.read_csv('C:\\Users\\Shyam\\Desktop\\pythonprojects\\datasets\\test-data.csv')

        print(train_data.head())

        # shape of the dataset
        print('Shape of training data :', train_data.shape)
        print('Shape of testing data :', test_data.shape)

        # Now, we need to predict the missing target variable in the test data
        # target variable - Survived

        # seperate the independent and target variable on training data
        train_x = train_data.drop(columns=['Survived'], axis=1)
        train_y = train_data['Survived']

        # seperate the independent and target variable on testing data
        test_x = test_data.drop(columns=['Survived'], axis=1)
        test_y = test_data['Survived']

        model = LogisticRegression()

        # fit the model with the training data
        model.fit(train_x, train_y)

        # coefficeints of the trained model
        print('Coefficient of model :', model.coef_)

        # intercept of the model
        print('Intercept of model', model.intercept_)

        # predict the target on the train dataset
        predict_train = model.predict(train_x)
        print('Target on train data', predict_train)

        # Accuray Score on train dataset
        accuracy_train = accuracy_score(train_y, predict_train)
        print('accuracy_score on train dataset : ', accuracy_train)

        # predict the target on the test dataset
        predict_test = model.predict(test_x)
        print('Target on test data', predict_test)

        # Accuracy Score on test dataset
        accuracy_test = accuracy_score(test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    elif text.lower() == 'balc' and text_1 != text.lower():
        print("Chose one to demonstrate:")
        print("1. Airbnb Prediction")
        print("2. Seattle Police PD Prediction")
        print("3. Text Analytics UTD")
        print("4. Image Classification")

        test = telegram_bot_sendtext("Chose one to demonstrate:")
        test = telegram_bot_sendtext("1. Airbnb Prediction")
        test = telegram_bot_sendtext("2. Seattle Police PD Prediction")
        test = telegram_bot_sendtext("3. Text Analytics UTD")
        test = telegram_bot_sendtext("4. Image Classification")
        text, chat = get_last_chat_id_and_text(get_updates())
        text_1 = 'balc'

    elif text.lower() == '1' and text_1 != text.lower():
        print("Doing Airbnb")
        test = telegram_bot_sendtext("1. Airbnb Prediction")
        text_1 = '1'

        df = pd.read_csv('AB_NYC_2019.csv')
        df.head()
        df2 = pd.get_dummies(df['room_type'], drop_first=True)
        df['Private room'] = d
        f2['Private room']
        df['Shared room'] = df2['Shared room']

        df3 = pd.get_dummies(df['neighbourhood_group'], drop_first=True)
        df['Brooklyn'] = df3['Brooklyn']
        df['Manhattan'] = df3['Manhattan']
        df['Queens'] = df3['Queens']
        df['Staten Island'] = df3['Staten Island']

        # Drop id and host id in new df1

        df1 = df.drop(
            ['id', 'host_id', 'name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review'],
            axis=1)

        # Co-relation matrix
        corr = df1.corr()
        corr.style.background_gradient(cmap='coolwarm')

        train_data = df1.loc[:20000]
        test_data = df1.loc[20000:]

        # Predict price using Linear Regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        print(train_data.head())

        # shape of the dataset
        print('\nShape of training data :', train_data.shape)
        print('\nShape of testing data :', test_data.shape)

        # Now, we need to predict the missing target variable in the test data
        # target variable - Item_Outlet_Sales

        # seperate the independent and target variable on training data
        train_x = train_data.drop(columns=['price'], axis=1)
        train_y = train_data['price']

        # seperate the independent and target variable on training data
        test_x = test_data.drop(columns=['price'], axis=1)
        test_y = test_data['price']

        '''
        Create the object of the Linear Regression model
        You can also add other parameters and test your code here
        Some parameters are : fit_intercept and normalize
        Documentation of sklearn LinearRegression: 

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

         '''
        model = LinearRegression()

        # fit the model with the training data
        model.fit(train_x, train_y)

        # coefficeints of the trained model
        print('\nCoefficient of model :', model.coef_)

        # intercept of the model
        print('\nIntercept of model', model.intercept_)

        # predict the target on the test dataset
        predict_train = model.predict(train_x)
        print('\nItem_Outlet_Sales on training data', predict_train)

        # Root Mean Squared Error on training dataset
        rmse_train = mean_squared_error(train_y, predict_train) ** (0.5)
        print('\nRMSE on train dataset : ', rmse_train)

        # predict the target on the testing dataset
        predict_test = model.predict(test_x)
        print('\nItem_Outlet_Sales on test data', predict_test)

        # Root Mean Squared Error on testing dataset
        rmse_test = mean_squared_error(test_y, predict_test) ** (0.5)
        print('\nRMSE on test dataset : ', rmse_test)



    ########################################################################################################
    ######################################SEATTLE PD##################################################
    ##################################################################################################

    elif text.lower() == '2' and text_1 != text.lower():
        print("Doing Seattle PD. Please Wait")
        test = telegram_bot_sendtext("2. Doing Seattle PD. Please Wait")
        text_1 = '2'






    ##################################################################################################
    ####################################TEXT ANALYTICS################################################
    ##################################################################################################

    elif text.lower() == '3' and text_1 != text.lower():
        print("Doing Text Analytics")
        test = telegram_bot_sendtext("3. Doing Text Analytics. Please Wait")
        text_1 = '3'







    ##################################################################################################
    ########################################IMAGE CLASSIFICATION######################################
    ##################################################################################################

    elif text.lower() == '4' and text_1 != text.lower():
        print("Doing Image Classification")
        test = telegram_bot_sendtext("4. Doing Image Classification. Please Wait")
        text_1 = '4'