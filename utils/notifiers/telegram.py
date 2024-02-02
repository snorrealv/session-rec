from telegram.ext.updater import Updater
from telegram.ext.commandhandler import CommandHandler
from utils.utils import assert_env_variables_set
import telegram
import socket
import traceback
import os


class Telegram():
    def __init__(self) -> None:
        assert_env_variables_set(['TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID'])

        self.token = os.environ.get('TELEGRAM_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.updater = Updater(self.token)  # , use_context=True
        self.updater.start_polling()
        self.bot = telegram.Bot(token=self.token)
        self.updater.dispatcher.add_handler( CommandHandler('status', self.status) )

    def status(self, bot, update):
        update.message.reply_text(
            'Running on {}'.format(socket.gethostname()))
        pass

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def send_exception(self, message):
        tmpfile = open('exception.txt', 'w')
        traceback.print_exc(file=tmpfile)
        tmpfile.close()
        self.send_file('exception.txt')
        self.send_message(message)

    def send_results(self, message):
        
        self.send_message(message)
