"""
Author: Harshavardhan
"""
from telegram.ext import Updater, InlineQueryHandler, CommandHandler, MessageHandler, Filters, ConversationHandler
import telegram
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
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
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
import smtplib

# GLOBAL VARIABLES
BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'
BOT_CHAT_ID = None  # Bot chat ID is no longer needed for starting the bot. Its dynamically picked

# MODEL RELATED
MODEL = None
PREPROCESS_INPUT = None
GRAPH = None
SESS = None
INIT = None
MODEL_TO_FETCH = 'nasnetlarge'

# EMAIL RELATED
SMTP_HOST = 'smtp-mail.outlook.com'
SMTP_PORT = 587
CON = None
FROM_ADDRESS = 'maniakbot@outlook.com'
FROM_ADDRESS_PASS = 'zasxcdfv20'
RECIPIENT_ADDRESS = None
SUBJECT = None
MESSAGE = None
ATTACHMENTS = None

# LOGGING
logging.root.removeHandler(absl.logging._absl_handler)
logfile_handler = logging.FileHandler('./bot.log')

absl.logging._warn_preinit_stderr = False
logging_format = '%(asctime)s - %(levelname)s - %(filename)s @%(lineno)d : %(funcName)s() - %(message)s'
logging_config = {
    'level': logging.INFO,
    'format': logging_format
}

logging.basicConfig(**logging_config)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(logging_format)
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def login_email():
    logger.info('Logging into mail server')
    global CON, SMTP_HOST, SMTP_PORT, FROM_ADDRESS, FROM_ADDRESS_PASS
    if not CON:
        import time
        start = time.time()
        CON = smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT)
        logger.info(f'Time taken to login is {time.time() - start} Secs')
        CON.starttls()
        success = CON.login(FROM_ADDRESS, FROM_ADDRESS_PASS)
        logger.info(f"Login {success}")
        return success
    return True


def msg_handler(update, context):
    text = update.message.text
    global BOT_CHAT_ID
    BOT_CHAT_ID = str(update.effective_chat.id)
    text_1 = 'a'
    if text == 'Lock' or text == 'lock':
        ctypes.windll.user32.LockWorkStation()

    elif text.lower() in ['hello', 'hey', 'hi', 'hola', 'yo', 'heylo']:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"{text.capitalize()}! My name is {update.effective_message.bot.first_name}")

    elif text.lower() == 'facebook' and text_1 != text.lower():
        wb.open('www.facebook.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Facebook")

    elif text.lower() == 'google' and text_1 != text.lower():
        wb.open('www.google.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Google")

    elif text.lower() == 'youtube' and text_1 != text.lower():
        wb.open('www.youtube.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Youtube")

    elif text.lower() == 'github' and text_1 != text.lower():
        wb.open('www.github.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Github")

    elif text.lower() == 'quora' and text_1 != text.lower():
        wb.open('www.quora.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened Quora")

    elif text.lower() == 'linkedin' and text_1 != text.lower():
        wb.open('www.linkedin.com')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened LinkedIn")

    elif text.lower() == 'elearning' and text_1 != text.lower():
        wb.open('https://elearning.utdallas.edu/webapps/portal/execute/tabs/tabAction?tab_tab_group_id=_1_1')
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Opened E-learning")

    elif text.lower() == 'balc' and text_1 != text.lower():
        logger.info("Chose one to demonstrate:")
        logger.info("1. Airbnb Prediction")
        logger.info("2. Seattle Police PD Prediction")
        logger.info("3. Text Analytics UTD")
        logger.info("4. Image Classification")

        update.message.reply_text(
            'Choose one for a demo:\n\n'
            '/airbnb_prediction\n\n'
            '/seattle_police_prediction\n\n'
            '/text_analytics\n\n'
            '/image_classification\n\n')


def get_np_array(image_bytes):
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return decoded


def classify(image_array):
    global MODEL, PREPROCESS_INPUT, GRAPH, SESS, INIT
    try:
        with GRAPH.as_default():
            set_session(SESS)
            # prepare image
            x = image_array
            # x = resize(x, (224, 224)) * 255
            # x = resize(x, (299, 299)) * 255
            x = resize(x, (331, 331)) * 255  # cast back to 0-255 range
            x = PREPROCESS_INPUT(x)
            x = np.expand_dims(x, 0)
            # processing image
            y = MODEL.predict(x)
            return decode_predictions(y)
    except Exception as e:
        logger.error(e)
        return None


def img_handler(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'll tell you what the image is in a moment.")
    photo_list = update.message['photo']
    file_id = photo_list[len(photo_list) - 1]['file_id']
    file_path = context.bot.getFile(file_id).file_path
    file = telegram.File(file_id, bot=context.bot, file_path=file_path)
    image_bytes = file.download_as_bytearray()
    image_np_array = get_np_array(image_bytes)
    predictions = classify(image_np_array)[0]
    if not predictions:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Couldn't figure that out. Try again later")
        return
    logger.info(predictions)
    best_pred = ' '.join([word.capitalize() for word in predictions[0][1].replace('_', ' ').split()])
    response = f'This is a {best_pred}. I am {predictions[0][2]*100:.2f}% confident'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Sending E-mail cancelled',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def send_to(update, context):
    update.message.reply_text(
        'Enter the E-mail address of the recipient')

    return RECIPIENT


def to_address(update, context):
    global RECIPIENT_ADDRESS
    RECIPIENT_ADDRESS = update.message.text
    update.message.reply_text(
        'Enter the E-mail subject')

    return ASK_SUBJECT


def email_subject(update, context):
    global SUBJECT
    SUBJECT = update.message.text
    update.message.reply_text(
        'Enter the E-mail message')
    return ASK_MESSAGE


def email_message(update, context):
    global MESSAGE
    MESSAGE = update.message.text
    # if send_email():
    #     update.message.reply_text(
    #         'Email sent successfully!')
    # else:
    #     update.message.reply_text(
    #         'Failed to send message')
    reply_keyboard = [['Yes', 'No']]
    update.message.reply_text(
            'Do you want to attach some files?', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return ASK_ATTACHMENTS


def ask_attachments(update, context):
    reply = update.message.text
    if reply == 'Yes':
        update.message.reply_text(
            'Send attachments')
        return GET_ATTACHMENTS
    else:
        if send_email():
            update.message.reply_text(
                'Email sent successfully')
        else:
            update.message.reply_text('Sending email failed')
        return ConversationHandler.END  # Ends conversation


def email_attachments(update, context):
    global ATTACHMENTS
    types = ['photo', 'audio', 'document', 'video']
    file_types_found = []
    file_bytes_list = {}
    for t in types:
        if update.message[t]:
            file_types_found.append(t)
    for file_type in file_types_found:
        if file_type == 'photo':
            file_id = update.message[file_type][len(update.message[file_type])-1]['file_id']
            file_name = update.message[file_type][len(update.message[file_type])-1]['file_id'] + '.jpg'
        else:
            file_id = update.message[file_type]['file_id']
            file_name = update.message[file_type]['file_name']
        file_path = context.bot.getFile(file_id).file_path
        file = telegram.File(file_id, bot=context.bot, file_path=file_path)
        file_bytes = bytes(file.download_as_bytearray())
        file_bytes_list[file_name] = file_bytes
    if send_email(files=file_bytes_list):
        update.message.reply_text(
            'Email sent successfully')
    else:
        update.message.reply_text(
            'Sending email failed')
    return ConversationHandler.END


def send_email(files: 'dict' = None) -> 'bool':
    global FROM_ADDRESS, RECIPIENT_ADDRESS, SUBJECT, MESSAGE, CON
    msg = MIMEMultipart()
    msg['From'] = FROM_ADDRESS
    msg['To'] = RECIPIENT_ADDRESS
    msg['Subject'] = SUBJECT
    msg.attach(MIMEText(MESSAGE, 'plain'))
    if files:
        for filename, file in files.items():
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            msg.attach(part)
    try:
        CON.send_message(msg)
        CON.quit()
        return True
    except Exception as e:
        logger.info(e)
        return False


RECIPIENT, ASK_SUBJECT, ASK_MESSAGE, ASK_ATTACHMENTS, GET_ATTACHMENTS, = range(5)

# COMMAND HANDLERS FROM HERE


def airbnb_prediction(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Doing airbnb prediction")
    # TODO


def seattle_police_prediction(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Doing Seattle PD prediction")
    # TODO


def text_analytics(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Doing Text analytics")
    # TODO


def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    email_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.regex('^(email|Email|E-mail|e-mail)$'), send_to)], # Triggers sending an Email conversation

        states={
            RECIPIENT: [MessageHandler(Filters.regex("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$)"), to_address)],

            ASK_SUBJECT: [MessageHandler(Filters.text, email_subject)],

            ASK_MESSAGE: [MessageHandler(Filters.text, email_message)],

            ASK_ATTACHMENTS: [MessageHandler(Filters.text, ask_attachments)],

            GET_ATTACHMENTS: [MessageHandler(Filters.document | Filters.photo | Filters.video | Filters.audio | Filters.contact, email_attachments)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Add all handlers to the dispatcher
    dp.add_handler(email_conv_handler)  # Email conversation handler
    dp.add_handler(MessageHandler(Filters.text, msg_handler))   # General text handler for greetings and BALC options
    dp.add_handler(MessageHandler(Filters.photo, img_handler))  # Handler to filter photos for image classification
    dp.add_handler(CommandHandler('airbnb_prediction', airbnb_prediction))
    dp.add_handler(CommandHandler('seattle_police_prediction', seattle_police_prediction))
    dp.add_handler(CommandHandler('text_analytics', text_analytics))
    dp.add_error_handler(error)
    logger.info('Polling for updates. Start chatting')
    updater.start_polling()
    updater.idle()


def load_model():
    global MODEL, PREPROCESS_INPUT, GRAPH, SESS, INIT, MODEL_TO_FETCH
    logger.info(f'Loading model {MODEL_TO_FETCH}')
    SESS = tf.Session()
    GRAPH = tf.get_default_graph()
    set_session(SESS)
    # INIT = tf.global_variables_initializer()
    init_model, PREPROCESS_INPUT = Classifiers.get(MODEL_TO_FETCH)
    # MODEL = init_model(input_shape=(224, 224, 3), weights='imagenet', classes=1000)
    # MODEL = init_model(input_shape=(299, 299, 3), weights='imagenet', classes=1000)
    MODEL = init_model(input_shape=(331, 331, 3), weights='imagenet', classes=1000)
    logger.info(f'Loaded model {MODEL_TO_FETCH}')


if __name__ == '__main__':
    logger.info('Clearing keras session')
    keras.backend.clear_session()
    login_email()
    load_model()
    main()
