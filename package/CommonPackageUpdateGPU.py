import os , shutil
import sys

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv-gpu/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = "pip install --no-index --find-links=venv-gpu/pip" if isLocalDir == True else "pip install" if isInstall == True else "pip download -d venv-gpu/pip"

# Hadoop套件 -------------------------------------------------
os.system("{} -m {} impyla==0.18.0".format(__pythonexePath,__pipFunction))          # 用於讀取Hive資料
os.system("{} -m {} hdfs==2.7.0".format(__pythonexePath,__pipFunction))             # 用於讀取HDFS資料
os.system("{} -m {} pyspark==3.3.1".format(__pythonexePath,__pipFunction))          # pyspark套件
os.system("{} -m {} py4j==0.10.9.5".format(__pythonexePath,__pipFunction))          # pyspark相依套件
# 機器學習套件 -------------------------------------------------
os.system("{} -m {} scikit-learn==1.1.3".format(__pythonexePath,__pipFunction))     # 機器學習套件
os.system("{} -m {} scikit-surprise==1.1.3".format(__pythonexePath,__pipFunction))  # 機器學習套件
os.system("{} -m {} mlxtend==0.21.0".format(__pythonexePath,__pipFunction))         # 機器學習套件
os.system("{} -m {} networkx==2.8.8".format(__pythonexePath,__pipFunction))         # 網路分析
os.system("{} -m {} xgboost==1.6.2".format(__pythonexePath,__pipFunction))          # 梯度提升樹
os.system("{} -m {} catboost==1.0.6".format(__pythonexePath,__pipFunction))         # 梯度提升樹
os.system("{} -m {} lightgbm==3.3.1".format(__pythonexePath,__pipFunction))         # 梯度提升樹
os.system("{} -m {} m2cgen==0.10.0".format(__pythonexePath,__pipFunction))          # 模型轉換套件
os.system("{} -m {} evidently==0.2.0".format(__pythonexePath,__pipFunction))        # 模型評估套件
# 自然語言套件 -------------------------------------------------
os.system("{} -m {} nltk==3.8".format(__pythonexePath,__pipFunction))               # 自然語言處理套件
os.system("{} -m {} rake_nltk==1.0.6".format(__pythonexePath,__pipFunction))        # 自然語言處理套件
os.system("{} -m {} gensim==4.2.0".format(__pythonexePath,__pipFunction))           # 自然語言處理套件
# Pytorch套件 -------------------------------------------------
os.system("{} -m {} torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu116".format(__pythonexePath,__pipFunction))           # 深度學習套件
os.system("{} -m {} torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu116".format(__pythonexePath,__pipFunction))     # 深度學習套件
os.system("{} -m {} torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116".format(__pythonexePath,__pipFunction))      # 深度學習套件
# Tensorflow 套件 -------------------------------------------------
os.system("{} -m {} tensorflow-gpu==2.12.0".format(__pythonexePath,__pipFunction))
# 自動機器學習套件 -------------------------------------------------
os.system("{} -m {} pycaret==3.0.0rc2".format(__pythonexePath,__pipFunction))       # 自動機器學習套件
os.system("{} -m {} autokeras==1.0.20".format(__pythonexePath,__pipFunction))       # 自動機器學習套件
# 其他套件 -------------------------------------------------
os.system("{} -m {} gym==0.26.2".format(__pythonexePath,__pipFunction))             # 遊戲場套件