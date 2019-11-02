"""
Author: Harshavardhan
"""
from telegram.ext import Updater, InlineQueryHandler, CommandHandler, MessageHandler, Filters
import telegram
import requests
import logging
import os
import cv2
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.backend import set_session
from classification_models.keras import Classifiers
import keras
import ctypes
import webbrowser as wb
import numpy as np
import tensorflow as tf
import absl.logging


#GLOBAL VARIABLES
bot_token = '938059809:AAGbZ0bub6QM-LoRaesWR585TEOUxw1w_YY'
bot_chatID = '971543311'
model = None
preprocess_input = None
graph = None
sess = None
init = None
model_to_fetch = 'nasnetlarge'

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging_config = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(filename)s @%(lineno)d : %(funcName)s() - %(message)s'
}
logging.basicConfig(**logging_config)


def telegram_bot_sendtext(bot_message):
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()


def cmd_handler(update, context):
    pass


def msg_handler(update, context):
    text = update.message.text
    global bot_chatID
    bot_chatID = str(update.effective_chat.id)
    text_1 = 'a'
    if text == 'Lock' or text == 'lock':
        ctypes.windll.user32.LockWorkStation()

    elif text.lower() in ['hello', 'hey', 'hi', 'hola', 'yo']:
        test = telegram_bot_sendtext(f"{text}! My name is {update.effective_message.bot.first_name}")
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
        pass


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

        telegram_bot_sendtext("Chose one to demonstrate:")
        telegram_bot_sendtext("1. Airbnb Prediction")
        telegram_bot_sendtext("2. Seattle Police PD Prediction")
        telegram_bot_sendtext("3. Text Analytics UTD")
        telegram_bot_sendtext("4. Image Classification")
        # text, chat = get_last_chat_id_and_text(get_updates())
        text_1 = 'balc'

    elif text.lower() == '1' and text_1 != text.lower():
        print("Doing Airbnb")
        test = telegram_bot_sendtext("1. Airbnb Prediction")
        text_1 = '1'

        # df = pd.read_csv('AB_NYC_2019.csv')
        # df.head()
        # df2 = pd.get_dummies(df['room_type'], drop_first=True)
        # df['Private room'] = d
        # f2['Private room']
        # df['Shared room'] = df2['Shared room']
        # import
        # df3 = pd.get_dummies(df['neighbourhood_group'], drop_first=True)
        # df['Brooklyn'] = df3['Brooklyn']
        # df['Manhattan'] = df3['Manhattan']
        # df['Queens'] = df3['Queens']
        # df['Staten Island'] = df3['Staten Island']
        #
        # # Drop id and host id in new df1
        #
        # df1 = df.drop(
        #     ['id', 'host_id', 'name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review'],
        #     axis=1)
        #
        # # Co-relation matrix
        # corr = df1.corr()
        # corr.style.background_gradient(cmap='coolwarm')
        #
        # train_data = df1.loc[:20000]
        # test_data = df1.loc[20000:]
        #
        # # Predict price using Linear Regression
        # from sklearn.linear_model import LinearRegression
        # from sklearn.metrics import mean_squared_error
        #
        # print(train_data.head())
        #
        # # shape of the dataset
        # print('\nShape of training data :', train_data.shape)
        # print('\nShape of testing data :', test_data.shape)
        #
        # # Now, we need to predict the missing target variable in the test data
        # # target variable - Item_Outlet_Sales
        #
        # # seperate the independent and target variable on training data
        # train_x = train_data.drop(columns=['price'], axis=1)
        # train_y = train_data['price']
        #
        # # seperate the independent and target variable on training data
        # test_x = test_data.drop(columns=['price'], axis=1)
        # test_y = test_data['price']
        #
        # '''
        # Create the object of the Linear Regression model
        # You can also add other parameters and test your code here
        # Some parameters are : fit_intercept and normalize
        # Documentation of sklearn LinearRegression:
        #
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        #
        #  '''
        # model = LinearRegression()
        #
        # # fit the model with the training data
        # model.fit(train_x, train_y)
        #
        # # coefficeints of the trained model
        # print('\nCoefficient of model :', model.coef_)
        #
        # # intercept of the model
        # print('\nIntercept of model', model.intercept_)
        #
        # # predict the target on the test dataset
        # predict_train = model.predict(train_x)
        # print('\nItem_Outlet_Sales on training data', predict_train)
        #
        # # Root Mean Squared Error on training dataset
        # rmse_train = mean_squared_error(train_y, predict_train) ** (0.5)
        # print('\nRMSE on train dataset : ', rmse_train)
        #
        # # predict the target on the testing dataset
        # predict_test = model.predict(test_x)
        # print('\nItem_Outlet_Sales on test data', predict_test)
        #
        # # Root Mean Squared Error on testing dataset
        # rmse_test = mean_squared_error(test_y, predict_test) ** (0.5)
        # print('\nRMSE on test dataset : ', rmse_test)



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
        test = telegram_bot_sendtext("4. Doing Image Classification. Send me a picture")
        text_1 = '4'


def get_np_array(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return decoded


def classify(image_array):
    global model, preprocess_input, graph, sess, init
    with graph.as_default():
        set_session(sess)
        # prepare image
        x = image_array
        # x = resize(x, (224, 224)) * 255
        # x = resize(x, (299, 299)) * 255
        x = resize(x, (331, 331)) * 255  # cast back to 0-255 range
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        # processing image
        y = model.predict(x)
        return decode_predictions(y)


def img_handler(update, context):
    telegram_bot_sendtext("I'll tell you what the image is in a moment.")
    photo_list = update.message['photo']
    file_id = photo_list[len(photo_list) - 1]['file_id']
    file_path = context.bot.getFile(file_id).file_path
    file = telegram.File(file_id, bot=context.bot, file_path=file_path)
    image_bytes = file.download_as_bytearray()
    image_np_array = get_np_array(image_bytes)
    predictions = classify(image_np_array)[0]
    print(predictions)
    best_pred = ' '.join([word.capitalize() for word in predictions[0][1].replace('_', ' ').split()])
    response = f'This is a {best_pred}. I am {predictions[0][2]*100:.2f}% confident'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

def main():
    updater = Updater(token=bot_token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, msg_handler))
    dp.add_handler(MessageHandler(Filters.photo, img_handler))
    updater.start_polling()
    updater.idle()

def load_model():
    global model, preprocess_input, graph, sess, init, model_to_fetch
    logging.info(f'Loading model {model_to_fetch}')
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    # init = tf.global_variables_initializer()
    init_model, preprocess_input = Classifiers.get(model_to_fetch)
    # model = init_model(input_shape=(224, 224, 3), weights='imagenet', classes=1000)
    # model = init_model(input_shape=(299, 299, 3), weights='imagenet', classes=1000)
    model = init_model(input_shape=(331, 331, 3), weights='imagenet', classes=1000)
    logging.info(f'Loaded model {model_to_fetch}')


if __name__ == '__main__':
    logging.info('Clearing session')
    keras.backend.clear_session()
    load_model()
    main()