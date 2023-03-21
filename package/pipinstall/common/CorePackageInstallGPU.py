import os , shutil
import sys

# 以 Python 3.9.13 為最該系統的版本

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv-gpu/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv-gpu/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = ("pip install --no-index --find-links=venv-gpu/pip" if isLocalDir == True else "pip install") if isInstall == True else "pip download -d venv-gpu/pip"

# Python套件 -------------------------------------------
os.system("{} -m {} pip==22.3.1".format(__pythonexePath,__pipFunction))
os.system("{} -m {} setuptools==65.6.3".format(__pythonexePath,__pipFunction))
# 基本套件 SSH套件 -------------------------------------------
os.system("{} -m {} gitpython==3.1.27".format(__pythonexePath,__pipFunction))
os.system("{} -m {} python-dotenv==0.21.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} paramiko==2.11.0".format(__pythonexePath,__pipFunction))
# 資料處理套件 -----------------------------------------------
os.system("{} -m {} pandas==1.4.4".format(__pythonexePath,__pipFunction))
os.system("{} -m {} numpy==1.22.4".format(__pythonexePath,__pipFunction))
os.system("{} -m {} scipy==1.8.1".format(__pythonexePath,__pipFunction))
# 資料庫套件 -------------------------------------------------
os.system("{} -m {} pyodbc==4.0.34".format(__pythonexePath,__pipFunction))
os.system("{} -m {} SQLAlchemy==1.4.41".format(__pythonexePath,__pipFunction))
os.system("{} -m {} {}==2.9.3".format(__pythonexePath,__pipFunction ,"psycopg2-binary" if os.name == "posix" else "psycopg2"))
# GoogleAPI套件 -------------------------------------------------
os.system("{} -m {} google-auth==2.15.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} oauth2client==4.1.3".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-auth-httplib2==0.1.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-auth-oauthlib==0.4.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-api-python-client==2.31.0".format(__pythonexePath,__pipFunction))
# 其他套件 -------------------------------------------------
os.system("{} -m {} matplotlib==3.6.0".format(__pythonexePath,__pipFunction))           # 繪圖套件
os.system("{} -m {} seaborn==0.12.1".format(__pythonexePath,__pipFunction))             # 繪圖套件
os.system("{} -m {} Pillow==9.3.0".format(__pythonexePath,__pipFunction))               # 圖片處理套件
os.system("{} -m {} Flask==2.2.2".format(__pythonexePath,__pipFunction))                # 網頁套件
os.system("{} -m {} streamlit==1.16.0".format(__pythonexePath,__pipFunction))           # 網頁套件
os.system("{} -m {} tqdm==4.64.1".format(__pythonexePath,__pipFunction))                # 進度條套件
os.system("{} -m {} openpyxl==3.0.10".format(__pythonexePath,__pipFunction))            # Excel套件


