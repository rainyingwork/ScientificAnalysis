import os , copy
import Config
from dotenv import load_dotenv
import OPSCommon as executeOPSCommon

load_dotenv(dotenv_path="env/postgresql.env")
load_dotenv(dotenv_path="env/mongodb.env")

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["ScientificAnalysisSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["DockerSet1_0_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0","D1_2_0","D1_2_1","D1_2_2","D1_3_0"]
        , "RepOPSRecordId": 9999
        , "RepFunctionArr": ["D1_1_0","D1_3_0"]
        , "RunFunctionArr": ["D1_2_0","D1_2_1","D1_2_2"]
        , "OrdFunctionArr": [
            {"Parent": "D1_2_0", "Child": "D1_2_1"}
            , {"Parent": "D1_2_1", "Child": "D1_2_2"}
        ]
        , "FunctionMemo": {
            "D1_1_0": "部署PostgreSQL"
            , "D1_2_0": "部署Python"
            , "D1_3_0": "部署MongoDB"
            , "D1_2_1": "安裝Python基本套件"
            , "D1_2_2": "安裝Python AI套件"
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
                            "/mfs/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data"
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
                    "python39": {
                        "image": "python:3.9.13-slim-bullseye"
                        , "restart": "always"
                        , "environment": {
                            "ACCEPT_EULA": "Y"
                        }
                        , "volumes": [
                            "/mfs/Docker/Python39/Volumes/Library:/Library"
                            , "/mfs/Docker/Python39/Volumes/Data:/Data"
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
                    "mongo": {
                        "image": "mongo:6.0.3"
                        , "restart": "always"
                        , "environment": {
                            "MONGO_INITDB_ROOT_USERNAME": os.getenv("POSTGRES_USERNAME")
                            , "MONGO_INITDB_ROOT_PASSWORD": os.getenv("POSTGRES_USERNAME")
                            , "MONGO_INITDB_DATABASE": "admin"
                        }
                        , "volumes": [
                            "/mfs/Docker/Mongo/Volumes/Data:/Data"
                        ]
                        , "ports": [
                            "27017:27017"
                        ]
                    }
                }
            }
        }
        , "D1_2_1": {
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
        , "D1_2_2": {
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
                , "docker exec -it python39 pip install torch==1.12.0"  # 深度學習套件
                , "docker exec -it python39 pip install torchvision==0.13.0"  # 深度學習套件
                , "docker exec -it python39 pip install torchaudio==0.13.0"  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                , "docker exec -it python39 pip install tensorflow==2.11.0"
                # 自動機器學習套件 -------------------------------------------------
                , "docker exec -it python39 pip install pycaret==3.0.0rc2" # 自動機器學習套件
                , "docker exec -it python39 pip install autokeras==1.0.20"  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                , "docker exec -it python39 pip install gym==0.26.2" # 遊戲場套件
            ]
        }

    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

