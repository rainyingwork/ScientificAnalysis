import os
from dotenv import load_dotenv
import requests

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

    def sendMessage(self, chatID=None, message=None):
        apiURL = 'https://api.telegram.org/bot{}/sendMessage'.format(self._allbottoken)
        try:
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
            return response.text
        except Exception as e:
            print(e)

    def sendPhoto(self, chatID=None, photo=None):
        apiURL = 'https://api.telegram.org/bot{}/sendPhoto'.format(self._allbottoken)
        try:
            response = requests.post(apiURL, {'chat_id': chatID}, files={'photo': open(photo, 'rb')})
            return response.text
        except Exception as e:
            print(e)

    def findMessage(self, offset=None,limit=100):
        api_url = 'https://api.telegram.org/bot{}/getUpdates'.format(self._allbottoken)
        try:
            response = requests.post(api_url)
            return response.text
        except Exception as e:
            print(e)


