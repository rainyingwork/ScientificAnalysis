import os , copy
from dotenv import load_dotenv
import OPSCommonLocal as executeOPSCommon

load_dotenv(dotenv_path="env/postgresql.env")

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"]
        , "Product": ["common"]
        , "Project": ["SciAnaSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        # "ExeFunctionArr": ["D1_1_0", "D1_1_1", "D1_1_2", "D1_2_0", "D1_2_1", "D1_2_2", "D1_6_0", "D2_1_0", "D2_2_0"],
        # "ExeFunctionArr": ["D1_2_0"],
        "ExeFunctionArr": ["D1_1_1","D1_1_2","D1_2_1", "D1_2_2"],
        # "RepOPSRecordId": 9999,
        # "RepFunctionArr": [],
        # "RunFunctionArr": ["D1_2_1", "D1_2_2"],
        "OrdFunctionArr": [
            #{"Parent": "D1_1_0", "Child": "D1_1_1"},
            {"Parent": "D1_1_1", "Child": "D1_1_2"},
            #{"Parent": "D1_2_0", "Child": "D1_2_1"},
            {"Parent": "D1_2_1", "Child": "D1_2_2"},
        ],
        "FunctionMemo": {
            "D1_1_0": "安裝Python39-CPU基底",
            "D1_1_1": "安裝Python39-CPU基本套件",
            "D1_1_2": "安裝Python39-CPU機器學習套件",
            "D1_2_0": "安裝Python39-GPU基底",
            "D1_2_1": "安裝Python39-GPU基本套件",
            "D1_2_2": "安裝Python39-GPU機器學習套件",
            "D1_6_0": "安裝PostgreSQL",
            "D2_1_0": "部署並重開Python39-CPU",
            "D2_2_0": "部署並重開Python39-GPU",
            "D2_6_0": "部署並重開PostgreSQL",
            "D2_8_0": "部署並重開Jenkins",
        },
    }
    opsInfo["ParameterJson"] = {
        "D1_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7" ,
                "services": {
                    "python39-cpu": {
                        "image": "ubuntu:22.04",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y" ,
                        },
                        "volumes": [
                            # "/Docker/Python39/Volumes/Library:/Library",
                            # "/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D1_1_1": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # 基本套件 --------------------------------------------------
                "docker exec -it python39-cpu pip install pip==23.1.2",
                "docker exec -it python39-cpu pip install setuptools==67.8.0",
                "docker exec -it python39-cpu pip install wheel==0.40.0",
                # 基本套件 SSH套件 -------------------------------------------
                "docker exec -it python39-cpu pip install gitpython==3.1.31",
                "docker exec -it python39-cpu pip install python-dotenv==1.0.0",
                "docker exec -it python39-cpu pip install paramiko==3.2.0",
                # 資料處理套件 -----------------------------------------------
                "docker exec -it python39-cpu pip install pandas==1.5.3",
                "docker exec -it python39-cpu pip install numpy==1.23.5",
                "docker exec -it python39-cpu pip install scipy==1.10.1",
                "docker exec -it python39-cpu pip install polars==0.18.2",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install pyodbc==4.0.39",
                "docker exec -it python39-cpu pip install SQLAlchemy==2.0.16",
                "docker exec -it python39-cpu pip install psycopg2-binary==2.9.6",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install google-auth==2.19.1",
                "docker exec -it python39-cpu pip install oauth2client==4.1.3",
                "docker exec -it python39-cpu pip install google-auth-httplib2==0.1.0",
                "docker exec -it python39-cpu pip install google-auth-oauthlib==1.0.0",
                "docker exec -it python39-cpu pip install google-api-python-client==2.89.0",
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install matplotlib==3.7.1",  # 繪圖套件
                "docker exec -it python39-cpu pip install seaborn==0.12.2",  # 繪圖套件
                "docker exec -it python39-cpu pip install Pillow==9.5.0",  # 圖片處理套件
                "docker exec -it python39-cpu pip install Flask==2.3.2",  # 網頁套件
                "docker exec -it python39-cpu pip install streamlit==1.23.1",  # 網頁套件
                "docker exec -it python39-cpu pip install requests==2.31.0",  # 進度條套件
                "docker exec -it python39-cpu pip install tqdm==4.65.0",  # 網頁套件
            ],
        },
        "D1_1_2": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install impyla==0.18.0",  # 用於讀取Hive資料
                "docker exec -it python39-cpu pip install hdfs==2.7.0",  # 用於讀取HDFS資料
                "docker exec -it python39-cpu pip install fastparquet==2023.4.0",  # 讀取parquet檔案
                # 機器學習套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install scikit-learn==1.2.2",  # 機器學習套件
                "docker exec -it python39-cpu pip install scikit-surprise==1.1.3",  # 機器學習套件
                "docker exec -it python39-cpu pip install mlxtend==0.22.0",  # 機器學習套件
                "docker exec -it python39-cpu pip install networkx==3.1",  # 網路分析
                "docker exec -it python39-cpu pip install xgboost==1.7.5",  # 梯度提升樹
                "docker exec -it python39-cpu pip install catboost==1.2",  # 梯度提升樹
                "docker exec -it python39-cpu pip install lightgbm==3.3.5",  # 梯度提升樹
                "docker exec -it python39-cpu pip install m2cgen==0.10.0",  # 模型轉換套件
                "docker exec -it python39-cpu pip install evidently==0.3.3",  # 模型評估套件
                "docker exec -it python39-cpu pip install tables==3.8.0",  # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install nltk==3.8.1",  # 自然語言處理套件
                "docker exec -it python39-cpu pip install rake_nltk==1.0.6",  # 自然語言處理套件
                "docker exec -it python39-cpu pip install gensim==4.3.1",  # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install torch==2.0.1",  # 深度學習套件
                "docker exec -it python39-cpu pip install torchvision==0.15.2",  # 深度學習套件
                "docker exec -it python39-cpu pip install torchaudio==2.0.2",  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install tensorflow==2.10.0",
                # 自動機器學習套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install pycaret==3.0.2",  # 自動機器學習套件
                "docker exec -it python39-cpu pip install autokeras==1.1.0",  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install gym==0.26.2",  # 遊戲場套件
            ],
        },
        "D1_2_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7" ,
                "services": {
                    "python39-gpu": {
                        "image": "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
                        "restart": "always" ,
                        "gpus": "all",
                        "environment": {
                            "ACCEPT_EULA": "Y" ,
                        },
                        "volumes": [
                            # "/Docker/Python39/Volumes/Library:/Library",
                            # "/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D1_2_1": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # 基本套件 --------------------------------------------------
                "docker exec -it python39-gpu pip install pip==23.1.2",
                "docker exec -it python39-gpu pip install setuptools==67.8.0",
                "docker exec -it python39-gpu pip install wheel==0.40.0",
                # 基本套件 SSH套件 -------------------------------------------
                "docker exec -it python39-gpu pip install gitpython==3.1.31",
                "docker exec -it python39-gpu pip install python-dotenv==1.0.0",
                "docker exec -it python39-gpu pip install paramiko==3.2.0",
                # 資料處理套件 -----------------------------------------------
                "docker exec -it python39-gpu pip install pandas==1.5.3",
                "docker exec -it python39-gpu pip install numpy==1.23.5",
                "docker exec -it python39-gpu pip install scipy==1.10.1",
                "docker exec -it python39-gpu pip install polars==0.18.2",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install pyodbc==4.0.39",
                "docker exec -it python39-gpu pip install SQLAlchemy==2.0.16",
                "docker exec -it python39-gpu pip install psycopg2-binary==2.9.6",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install google-auth==2.19.1",
                "docker exec -it python39-gpu pip install oauth2client==4.1.3",
                "docker exec -it python39-gpu pip install google-auth-httplib2==0.1.0",
                "docker exec -it python39-gpu pip install google-auth-oauthlib==1.0.0",
                "docker exec -it python39-gpu pip install google-api-python-client==2.89.0",
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install matplotlib==3.7.1",  # 繪圖套件
                "docker exec -it python39-gpu pip install seaborn==0.12.2",  # 繪圖套件
                "docker exec -it python39-gpu pip install Pillow==9.5.0",  # 圖片處理套件
                "docker exec -it python39-gpu pip install Flask==2.3.2",  # 網頁套件
                "docker exec -it python39-gpu pip install fastapi==0.97.0",  # 網頁套件
                "docker exec -it python39-gpu pip install uvicorn==0.22.0",  # 網頁套件
                "docker exec -it python39-gpu pip install streamlit==1.23.1",  # 網頁套件
                "docker exec -it python39-gpu pip install requests==2.31.0",  # 進度條套件
                "docker exec -it python39-gpu pip install tqdm==4.65.0",  # 網頁套件
            ],
        },
        "D1_2_2": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install impyla==0.18.0",  # 用於讀取Hive資料
                "docker exec -it python39-gpu pip install hdfs==2.7.0",  # 用於讀取HDFS資料
                "docker exec -it python39-gpu pip install fastparquet==2023.4.0",  # 讀取parquet檔案
                # 機器學習套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install scikit-learn==1.2.2",  # 機器學習套件
                "docker exec -it python39-gpu pip install scikit-surprise==1.1.3",  # 機器學習套件
                "docker exec -it python39-gpu pip install mlxtend==0.22.0",  # 機器學習套件
                "docker exec -it python39-gpu pip install networkx==3.1",  # 網路分析
                "docker exec -it python39-gpu pip install xgboost==1.7.5",  # 梯度提升樹
                "docker exec -it python39-gpu pip install catboost==1.2",  # 梯度提升樹
                "docker exec -it python39-gpu pip install lightgbm==3.3.5",  # 梯度提升樹
                "docker exec -it python39-gpu pip install m2cgen==0.10.0",  # 模型轉換套件
                "docker exec -it python39-gpu pip install evidently==0.3.3",  # 模型評估套件
                "docker exec -it python39-gpu pip install tables==3.8.0",  # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install nltk==3.8.1",  # 自然語言處理套件
                "docker exec -it python39-gpu pip install rake_nltk==1.0.6",  # 自然語言處理套件
                "docker exec -it python39-gpu pip install gensim==4.3.1",  # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118",  # 深度學習套件
                "docker exec -it python39-gpu pip install torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118",  # 深度學習套件
                "docker exec -it python39-gpu pip install torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118",  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install tensorflow==2.10.0",
                # 自動機器學習套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install pycaret==3.0.2",  # 自動機器學習套件
                "docker exec -it python39-gpu pip install autokeras==1.1.0",  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install gym==0.26.2",  # 遊戲場套件
            ],
        },
        "D1_6_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "postgresql": {
                        "image": "postgres:15.1",
                        "restart": "always",
                        "environment": {
                            "POSTGRES_DB": "postgres",
                            "POSTGRES_USER": os.getenv("POSTGRES_USERNAME"),
                            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
                            "PGDATA": "/lib/postgresql/data",
                        },
                        "volumes_clean": [
                            "/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data"
                            # 本機位置 : 遠端位置
                        ],
                        "ports": [
                            "5432:5432"
                        ],
                    },
                },
            },
        },
        "D2_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-basic": {
                        "image": "vicying/python:3.9.13-cpu-1.1.2",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes_clean": [
                            "/mfs/Docker/Python39/Volumes/Library:/Library",
                            "/mfs/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D2_2_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-master": {
                        "image": "vicying/python:3.9.13-cpu-1.1.2",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/mfs/Docker/Python39/Volumes/Library:/Library",
                            "/mfs/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D2_3_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-slave": {
                        "image": "vicying/python:3.9.13-gpu-1.1.2",
                        "restart": "always",
                        "gpus": "all",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/mfs/Docker/Python39/Volumes/Library:/Library",
                            "/mfs/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D2_6_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "postgresql": {
                        "image": "postgres:15.1",
                        "restart": "always",
                        "environment": {
                            "POSTGRES_DB": "postgres",
                            "POSTGRES_USER": os.getenv("POSTGRES_USERNAME"),
                            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
                            "PGDATA": "/lib/postgresql/data",
                        },
                        "volumes": [
                            "/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data",
                            # 本機位置 : 遠端位置
                        ],
                        "ports": [
                            "5432:5432",
                        ]
                    },
                },
            },
        },
        "D2_8_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "jenkins": {
                        "image": "jenkins/jenkins:lts-jdk11",
                        "restart": "always",
                        "privileged" : "true", # 要有root權限才可以安裝套件
                        "environment": {
                            "TZ":"Asia/Taipei" ,
                            "JAVA_OPTS":"-Duser.timezone=Asia/Taipei" ,
                        },
                        "volumes_clean": [
                            "/mfs/Docker/Jenkins/Volumes/jenkins_home:/var/jenkins_home",
                        ],
                        "volumes": [
                            "/etc/localtime:/etc/localtime",
                            "/var/run/docker.sock:/var/run/docker.sock",
                        ],
                        "ports": [
                            "8080:8080",
                            "50000:50000",
                        ],
                    },
                },
            },
        }, #  "image": "jenkinsci/blueocean",
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
