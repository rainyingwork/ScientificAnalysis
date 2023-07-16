import os , shutil
import sys

# 以 Python 3.9.13 為最該系統的版本

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = ("pip install --no-index --find-links=venv/pip" if isLocalDir == True else "pip install") if isInstall == True else "pip download -d venv/pip"

# Python套件 -------------------------------------------
os.system("{} -m {} pip==23.1.2".format(__pythonexePath,__pipFunction))
os.system("{} -m {} setuptools==67.8.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} wheel==0.40.0".format(__pythonexePath,__pipFunction))
# 基本套件 SSH套件 -------------------------------------------
os.system("{} -m {} gitpython==3.1.31".format(__pythonexePath,__pipFunction))
os.system("{} -m {} python-dotenv==1.0.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} paramiko==3.2.0".format(__pythonexePath,__pipFunction))
# 資料處理套件 -----------------------------------------------
os.system("{} -m {} pandas==1.5.3".format(__pythonexePath,__pipFunction))
os.system("{} -m {} numpy==1.23.5".format(__pythonexePath,__pipFunction))
os.system("{} -m {} scipy==1.10.1".format(__pythonexePath,__pipFunction))
os.system("{} -m {} polars==0.18.2".format(__pythonexePath,__pipFunction))
# 資料庫套件 -------------------------------------------------
os.system("{} -m {} pyodbc==4.0.39".format(__pythonexePath,__pipFunction))
os.system("{} -m {} SQLAlchemy==2.0.16".format(__pythonexePath,__pipFunction))
os.system("{} -m {} {}==2.9.6".format(__pythonexePath,__pipFunction ,"psycopg2-binary" if os.name == "posix" else "psycopg2"))
# GoogleAPI套件 -------------------------------------------------
os.system("{} -m {} google-auth==2.19.1".format(__pythonexePath,__pipFunction))
os.system("{} -m {} oauth2client==4.1.3".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-auth-httplib2==0.1.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-auth-oauthlib==1.0.0".format(__pythonexePath,__pipFunction))
os.system("{} -m {} google-api-python-client==2.89.0".format(__pythonexePath,__pipFunction))
# 其他套件 -------------------------------------------------
os.system("{} -m {} matplotlib==3.7.1".format(__pythonexePath,__pipFunction))           # 繪圖套件
os.system("{} -m {} seaborn==0.12.2".format(__pythonexePath,__pipFunction))             # 繪圖套件
os.system("{} -m {} Pillow==9.5.0".format(__pythonexePath,__pipFunction))               # 圖片處理套件
os.system("{} -m {} Flask==2.3.2".format(__pythonexePath,__pipFunction))                # 網頁套件
os.system("{} -m {} fastapi==0.97.0".format(__pythonexePath,__pipFunction))             # 網頁套件
os.system("{} -m {} uvicorn==0.22.0".format(__pythonexePath,__pipFunction))             # 網頁套件
os.system("{} -m {} streamlit==1.23.1".format(__pythonexePath,__pipFunction))           # 網頁套件
os.system("{} -m {} requests==2.31.0".format(__pythonexePath,__pipFunction))            # 網頁套件
os.system("{} -m {} tqdm==4.65.0".format(__pythonexePath,__pipFunction))                # 進度條套件
os.system("{} -m {} openpyxl==3.1.2".format(__pythonexePath,__pipFunction))             # Excel套件
os.system("{} -m {} selenium==4.10.0".format(__pythonexePath,__pipFunction))            # 爬蟲專用套件
os.system("{} -m {} beautifulsoup4==4.12.2".format(__pythonexePath,__pipFunction))      # 爬蟲網頁解析套件
# 安裝pygraphviz 請先至 graphviz 安裝2.46以上版本下載位置 https://graphviz.org/download/
os.system('{} -m {} --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz==1.11'.format(__pythonexePath,__pipFunction))

