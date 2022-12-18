import os , shutil
import sys

pythonEXEPath = "bin" if os.name == "posix" else "Scripts"
__pythonexePath = "{}/venv/{}/python".format(sys.path[1],pythonEXEPath)

# Hadoop套件 -------------------------------------------------
os.system("{} -m pip install impyla==0.18.0".format(__pythonexePath)) # 用於讀取Hive資料
os.system("{} -m pip install hdfs==2.7.0".format(__pythonexePath)) # 用於讀取HDFS資料
os.system("{} -m pip install pyspark==3.3.1".format(__pythonexePath)) # pyspark套件
os.system("{} -m pip install py4j==0.10.9.7".format(__pythonexePath)) # pyspark相依套件
# 機器學習套件 -------------------------------------------------
os.system("{} -m pip install scikit-learn==1.1.3".format(__pythonexePath)) # 機器學習套件
os.system("{} -m pip install scikit-surprise==1.1.3".format(__pythonexePath)) # 機器學習套件
os.system("{} -m pip install mlxtend==0.21.0".format(__pythonexePath)) # 機器學習套件
os.system("{} -m pip install networkx==2.8.8".format(__pythonexePath)) # 網路分析
os.system("{} -m pip install xgboost==1.6.2".format(__pythonexePath)) # 梯度提升樹
os.system("{} -m pip install catboost==1.0.6".format(__pythonexePath)) # 梯度提升樹
os.system("{} -m pip install lightgbm==3.3.1".format(__pythonexePath)) # 梯度提升樹
os.system("{} -m pip install m2cgen==0.10.0".format(__pythonexePath)) # 模型轉換套件
os.system("{} -m pip install evidently==0.2.0".format(__pythonexePath)) # 模型評估套件
# 自然語言套件 -------------------------------------------------
os.system("{} -m pip install nltk==3.8".format(__pythonexePath)) # 自然語言處理套件
os.system("{} -m pip install rake_nltk==1.0.6".format(__pythonexePath)) # 自然語言處理套件
os.system("{} -m pip install gensim==4.2.0".format(__pythonexePath)) # 自然語言處理套件
# Pytorch套件 -------------------------------------------------
os.system("{} -m pip install torch==1.12.0".format(__pythonexePath))  # 深度學習套件
os.system("{} -m pip install torchvision==0.13.0".format(__pythonexePath))  # 深度學習套件
os.system("{} -m pip install torchaudio==0.13.0".format(__pythonexePath))  # 深度學習套件
# Tensorflow 套件 -------------------------------------------------
os.system("{} -m pip install tensorflow==2.11.0".format(__pythonexePath))
# 自動機器學習套件 -------------------------------------------------
os.system("{} -m pip install pycaret==3.0.0rc2".format(__pythonexePath))# 自動機器學習套件
os.system("{} -m pip install autokeras==1.0.20".format(__pythonexePath))  # 自動機器學習套件
# 其他套件 -------------------------------------------------
os.system("{} -m pip install gym==0.26.2".format(__pythonexePath)) # 遊戲場套件