import os , shutil
import sys

# 以 Python 3.9.13 為最該系統的版本

pythonEXEPath = "bin" if os.name == "posix" else "Scripts"
__pythonexePath = "{}/venv/{}/python".format(sys.path[1],pythonEXEPath)

# # Python套件 -------------------------------------------
# os.system("{} -m pip install pip==22.3.1".format(__pythonexePath))
# os.system("{} -m pip install setuptools==65.6.3".format(__pythonexePath))
# # 基本套件 SSH套件 -------------------------------------------
# os.system("{} -m pip install gitpython==3.1.27".format(__pythonexePath))
# os.system("{} -m pip install python-dotenv==0.21.0".format(__pythonexePath))
# os.system("{} -m pip install paramiko==2.11.0".format(__pythonexePath))
# # 資料處理套件 -----------------------------------------------
# os.system("{} -m pip install pandas==1.4.4".format(__pythonexePath))
# os.system("{} -m pip install numpy==1.22.4".format(__pythonexePath))
# os.system("{} -m pip install scipy==1.8.1".format(__pythonexePath))
# # 資料庫套件 -------------------------------------------------
# os.system("{} -m pip install pyodbc==4.0.34".format(__pythonexePath))
# os.system("{} -m pip install SQLAlchemy==1.4.41".format(__pythonexePath))
# os.system("{} -m pip install {}==2.9.3".format(__pythonexePath ,"psycopg2-binary" if os.name == "posix" else "psycopg2"))
# # GoogleAPI套件 -------------------------------------------------
# os.system("{} -m pip install google-auth==2.15.0".format(__pythonexePath))
# os.system("{} -m pip install oauth2client==4.1.3".format(__pythonexePath))
# os.system("{} -m pip install google-auth-httplib2==0.1.0".format(__pythonexePath))
# os.system("{} -m pip install google-auth-oauthlib==0.4.0".format(__pythonexePath))
# os.system("{} -m pip install google-api-python-client==2.31.0".format(__pythonexePath))
# # 其他套件 -------------------------------------------------
# os.system("{} -m pip install matplotlib==3.6.0".format(__pythonexePath))  # 繪圖套件
# os.system("{} -m pip install seaborn==0.12.1".format(__pythonexePath))  # 繪圖套件
# os.system("{} -m pip install Pillow==9.3.0".format(__pythonexePath)) # 圖片處理套件
# os.system("{} -m pip install Flask==2.2.2".format(__pythonexePath)) # 網頁套件
# os.system("{} -m pip install streamlit==1.16.0".format(__pythonexePath)) # 網頁套件
# os.system("{} -m pip install tqdm==4.64.1".format(__pythonexePath)) # 進度條套件
# os.system("{} -m pip install openpyxl==3.0.10".format(__pythonexePath)) # Excel套件

if os.name == "posix" :
    os.system("{} -m pip install certifi==2022.12.7".format(__pythonexePath)) # 修正SSL錯誤


