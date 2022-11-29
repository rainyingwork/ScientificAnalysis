import os , sys

class ThredingCtrl:

    def __init__(self):
        pass

    def passThread(self):
        print("Error Pass Thread!!")
        sys.exit(0)

    def stopThread(self):
        print("Error Stop Thread!!")
        os._exit(0)


