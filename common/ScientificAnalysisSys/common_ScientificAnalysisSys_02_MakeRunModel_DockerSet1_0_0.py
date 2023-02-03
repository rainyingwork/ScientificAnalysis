import os , copy
import Config
from dotenv import load_dotenv
import OPSCommonLocal as executeOPSCommon

load_dotenv(dotenv_path="env/postgresql.env")
load_dotenv(dotenv_path="env/mongo.env")

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["common"]
        , "Project": ["ScientificAnalysisSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["DockerSet1_0_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0","D1_2_0","D1_3_0","D1_3_1","D1_3_2","D1_4_0","D1_4_1","D1_4_2"]
        # , "RepOPSRecordId": 9999
        # , "RepFunctionArr": []
        # , "RunFunctionArr": ["D1_3_1","D1_3_2"]
        , "OrdFunctionArr": [
            {"Parent": "D1_3_0", "Child": "D1_3_1"},
            {"Parent": "D1_3_1", "Child": "D1_3_2"},
            {"Parent": "D1_4_0", "Child": "D1_4_1"},
            {"Parent": "D1_4_1", "Child": "D1_4_2"},
        ]
        , "FunctionMemo": {
            "D1_1_0": "部署PostgreSQL"
            , "D1_2_0": "部署MongoDB"
            , "D1_3_0": "部署Python39"
            , "D1_3_1": "安裝Python39基本套件"
            , "D1_3_2": "安裝Python39機器學習套件"
            , "D1_4_0": "部署Python39-GPU"
            , "D1_4_1": "安裝Python39-GPU基本套件"
            , "D1_4_2": "安裝Python39-GPU機器學習套件"
        }
    }
    opsInfo["ParameterJson"] = {
        "D1_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo"
            , "DockerComposeInfo": {
                "version": "3.7"
                , "services": {
                    "postgresql": {
                        "image": "postgres:15.1"
                        , "restart": "always"
                        , "environment": {
                            "POSTGRES_DB": "postgres"
                            , "POSTGRES_USER": os.getenv("POSTGRES_USERNAME")
                            , "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD")
                            , "PGDATA": "/lib/postgresql/data"
                        }
                        , "volumes": [
                            "/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data"
                            # 本機位置 : 遠端位置
                        ]
                        , "ports": [
                            "5432:5432"
                        ]
                    }
                }
            }
        }
        , "D1_2_0": {
            "FunctionType": "RunContainerByDockerComposeInfo"
            , "DockerComposeInfo": {
                "version": "3.7"
                , "services": {
                    "mongo": {
                        "image": "mongo:6.0.3"
                        , "restart": "always"
                        , "environment": {
                            "MONGO_INITDB_ROOT_USERNAME": os.getenv("MONGO_USERNAME")
                            , "MONGO_INITDB_ROOT_PASSWORD": os.getenv("MONGO_PASSWORD")
                            , "MONGO_INITDB_DATABASE": "admin"
                        }
                        , "volumes": [
                            "/Docker/Mongo/Volumes/Data:/Data"
                        ]
                        , "ports": [
                            "27017:27017"
                        ]
                    }
                }
            }
        }
        , "D1_3_0": {
            "FunctionType": "RunContainerByDockerComposeInfo"
            , "DockerComposeInfo": {
                "version": "3.7"
                , "services": {
                    "python39": {
                        "image": "python:3.9.13-slim-bullseye"
                        , "restart": "always"
                        , "environment": {
                            "ACCEPT_EULA": "Y"
                        }
                        , "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library" ,
                            "/Docker/Python39/Volumes/Data:/Data" ,
                        ]
                    }
                }
            }
        }
        , "D1_3_1": {
            "FunctionType": "RunDockerCmdStr"
            , "DockerCmdStrs": [
                """
                "docker exec -it python39 apt update"
                 ,"docker exec -it python39 apt install -y build-essential"
                """
                # 基本套件 --------------------------------------------------
                "docker exec -it python39 pip install pip==22.3.1"
                , "docker exec -it python39 pip install setuptools==65.6.3"
                # 基本套件 SSH套件 -------------------------------------------
                , "docker exec -it python39 pip install gitpython==3.1.27"
                , "docker exec -it python39 pip install python-dotenv==0.21.0"
                , "docker exec -it python39 pip install paramiko==2.11.0"
                # 資料處理套件 -----------------------------------------------
                , "docker exec -it python39 pip install pandas==1.4.4"
                , "docker exec -it python39 pip install numpy==1.22.4"
                , "docker exec -it python39 pip install scipy==1.8.1"
                # 資料庫套件 -------------------------------------------------
                , "docker exec -it python39 pip install pyodbc==4.0.34"
                , "docker exec -it python39 pip install SQLAlchemy==1.4.41"
                , "docker exec -it python39 pip install psycopg2-binary==2.9.3"
                # 其他套件 -------------------------------------------------
                , "docker exec -it python39 pip install matplotlib==3.6.0"  # 繪圖套件
                , "docker exec -it python39 pip install seaborn==0.12.1"  # 繪圖套件
                , "docker exec -it python39 pip install pillow==9.3.0" # 圖片處理套件
                , "docker exec -it python39 pip install Flask==2.2.2" # 網頁套件
                , "docker exec -it python39 pip install streamlit==1.16.0" # 網頁套件
                , "docker exec -it python39 pip install tqdm==4.64.1" # 進度條套件
            ]
        }
        , "D1_3_2": {
            "FunctionType": "RunDockerCmdStr"
            , "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39 pip install impyla==0.18.0" # 用於讀取Hive資料
                , "docker exec -it python39 pip install hdfs==2.7.0" # 用於讀取HDFS資料
                , "docker exec -it python39 pip install pyspark==3.3.1" # pyspark套件
                , "docker exec -it python39 pip install py4j==0.10.9.7" # pyspark相依套件
                # 機器學習套件 -------------------------------------------------
                , "docker exec -it python39 pip install scikit-learn==1.1.3" # 機器學習套件
                , "docker exec -it python39 pip install scikit-surprise==1.1.3" # 機器學習套件
                , "docker exec -it python39 pip install mlxtend==0.21.0" # 機器學習套件
                , "docker exec -it python39 pip install networkx==2.8.8" # 網路分析
                , "docker exec -it python39 pip install xgboost==1.6.2" # 梯度提升樹
                , "docker exec -it python39 pip install catboost==1.0.6" # 梯度提升樹
                , "docker exec -it python39 pip install lightgbm==3.3.1" # 梯度提升樹
                , "docker exec -it python39 pip install m2cgen==0.10.0" # 模型轉換套件
                , "docker exec -it python39 pip install evidently==0.2.0" # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                , "docker exec -it python39 pip install nltk==3.8" # 自然語言處理套件
                , "docker exec -it python39 pip install rake_nltk==1.0.6" # 自然語言處理套件
                , "docker exec -it python39 pip install gensim==4.2.0" # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                , "docker exec -it python39 pip install --no-cache-dir torch==1.13.0"  # 深度學習套件
                , "docker exec -it python39 pip install --no-cache-dir torchvision==0.14.0"  # 深度學習套件
                , "docker exec -it python39 pip install --no-cache-dir torchaudio==0.13.0"  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                , "docker exec -it python39 pip install --no-cache-dir tensorflow==2.11.0"
                # 自動機器學習套件 -------------------------------------------------
                , "docker exec -it python39 pip install --no-cache-dir pycaret==3.0.0rc2" # 自動機器學習套件
                , "docker exec -it python39 pip install --no-cache-dir autokeras==1.0.20"  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                , "docker exec -it python39 pip install gym==0.26.2" # 遊戲場套件
            ]
        }
        , "D1_4_0": {
            "FunctionType": "RunContainerByDockerComposeInfo"
            , "DockerComposeInfo": {
                "version": "3.7"
                , "services": {
                    "python39-gpu": {
                        "image": "python:3.9.13-slim-bullseye"
                        , "restart": "always"
                        , "environment": {
                            "ACCEPT_EULA": "Y"
                        }
                        , "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library" ,
                            "/Docker/Python39/Volumes/Data:/Data" ,
                        ]
                    }
                }
            }
        }
        , "D1_4_1": {
            "FunctionType": "RunDockerCmdStr"
            , "DockerCmdStrs": [
                "docker exec -it python39-gpu apt update"
                ,"docker exec -it python39-gpu apt install -y build-essential"
                # 基本套件 --------------------------------------------------
                "docker exec -it python39-gpu pip install pip==22.3.1"
                , "docker exec -it python39-gpu pip install setuptools==65.6.3"
                # 基本套件 SSH套件 -------------------------------------------
                , "docker exec -it python39-gpu pip install gitpython==3.1.27"
                , "docker exec -it python39-gpu pip install python-dotenv==0.21.0"
                , "docker exec -it python39-gpu pip install paramiko==2.11.0"
                # 資料處理套件 -----------------------------------------------
                , "docker exec -it python39-gpu pip install pandas==1.4.4"
                , "docker exec -it python39-gpu pip install numpy==1.22.4"
                , "docker exec -it python39-gpu pip install scipy==1.8.1"
                # 資料庫套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install pyodbc==4.0.34"
                , "docker exec -it python39-gpu pip install SQLAlchemy==1.4.41"
                , "docker exec -it python39-gpu pip install psycopg2-binary==2.9.3"
                # 其他套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install matplotlib==3.6.0"  # 繪圖套件
                , "docker exec -it python39-gpu pip install seaborn==0.12.1"  # 繪圖套件
                , "docker exec -it python39-gpu pip install pillow==9.3.0" # 圖片處理套件
                , "docker exec -it python39-gpu pip install Flask==2.2.2" # 網頁套件
                , "docker exec -it python39-gpu pip install streamlit==1.16.0" # 網頁套件
                , "docker exec -it python39-gpu pip install tqdm==4.64.1" # 進度條套件
            ]
        }
        , "D1_4_2": {
            "FunctionType": "RunDockerCmdStr"
            , "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install impyla==0.18.0" # 用於讀取Hive資料
                , "docker exec -it python39-gpu pip install hdfs==2.7.0" # 用於讀取HDFS資料
                , "docker exec -it python39-gpu pip install pyspark==3.3.1" # pyspark套件
                , "docker exec -it python39-gpu pip install py4j==0.10.9.7" # pyspark相依套件
                # 機器學習套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install scikit-learn==1.1.3" # 機器學習套件
                , "docker exec -it python39-gpu pip install scikit-surprise==1.1.3" # 機器學習套件
                , "docker exec -it python39-gpu pip install mlxtend==0.21.0" # 機器學習套件
                , "docker exec -it python39-gpu pip install networkx==2.8.8" # 網路分析
                , "docker exec -it python39-gpu pip install xgboost==1.6.2" # 梯度提升樹
                , "docker exec -it python39-gpu pip install catboost==1.0.6" # 梯度提升樹
                , "docker exec -it python39-gpu pip install lightgbm==3.3.1" # 梯度提升樹
                , "docker exec -it python39-gpu pip install m2cgen==0.10.0" # 模型轉換套件
                , "docker exec -it python39-gpu pip install evidently==0.2.0" # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install nltk==3.8" # 自然語言處理套件
                , "docker exec -it python39-gpu pip install rake_nltk==1.0.6" # 自然語言處理套件
                , "docker exec -it python39-gpu pip install gensim==4.2.0" # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install --no-cache-dir torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu116"  # 深度學習套件
                , "docker exec -it python39-gpu pip install --no-cache-dir torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu116"  # 深度學習套件
                , "docker exec -it python39-gpu pip install --no-cache-dir torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116"  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install --no-cache-dir tensorflow-gpu==2.12.0 "
                # 自動機器學習套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install --no-cache-dir pycaret==3.0.0rc2" # 自動機器學習套件
                , "docker exec -it python39-gpu pip install --no-cache-dir autokeras==1.0.20"  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                , "docker exec -it python39-gpu pip install gym==0.26.2" # 遊戲場套件
            ]
        }

    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

