import os
import warnings ; warnings.filterwarnings('ignore')
import pandas
import threading
from package.common.osbasic.ThredingCtrl import ThredingCtrl

# threading.excepthook = ThredingCtrl.stopThread

pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 2000)
pandas.set_option('display.float_format', '{:,.5f}'.format)

class Config () :
    buildUsers = ['vicying']
    productNames = ['Example']
