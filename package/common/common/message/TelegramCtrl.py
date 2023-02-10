import os
from dotenv import load_dotenv
import telegram

class TelegramCtrl:

    def __init__(self, botuserid=None, bottoken=None, env=None):
        if env == None:
            self._botuserid = botuserid
            self._bottoken = bottoken
        else:
            load_dotenv(dotenv_path=env)
            self._botuserid = os.getenv("TELEGRAM_BOT_USERID")
            self._bottoken = os.getenv("TELEGRAM_BOT_TOKEN")
        self._allbottoken = self._botuserid + ":" + self._bottoken

    def sendMessage(self, chatid=None, massage=None):
        telegramBot = telegram.Bot(token=self._allbottoken)
        telegramBot.send_message(chat_id=chatid, text=massage)

    def sendPhoto(self, chatid=None, photo=None):
        telegramBot = telegram.Bot(token=self._allbottoken)
        telegramBot.send_photo(chat_id=chatid, photo=open(photo, 'rb'))

    def findMessage(self, offset=None,limit=100):
        telegramBot = telegram.Bot(token=self._allbottoken)
        return telegramBot.get_updates(offset=offset,limit=limit)
