import os
# ==================================================      ==================================================
import warnings ; warnings.filterwarnings('ignore')
import ssl ; ssl._create_default_https_context = ssl._create_unverified_context
# ==================================================      ==================================================
import matplotlib ; matplotlib.use('Agg')
# ==================================================      ==================================================
import threading
from package.common.osbasic.ThredingCtrl import ThredingCtrl
# threading.excepthook = ThredingCtrl.stopThread
# ==================================================      ==================================================
import pandas
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 2000)
pandas.set_option('display.float_format', '{:,.5f}'.format)
# ==================================================      ==================================================
class Config () :
    buildUsers = ['vicying']
    productNames = ['Example']
