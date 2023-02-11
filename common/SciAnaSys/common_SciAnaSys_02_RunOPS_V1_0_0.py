import os , copy
import OPSCommon as executeOPSCommon

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
        "ExeFunctionArr": ["D2_2_0"],
        # "RepOPSRecordId": 9999,
        # "RepFunctionArr": [],
        # "RunFunctionArr": ["D1_2_1", "D1_2_2"],
        "OrdFunctionArr": [
            # {"Parent": "D1_1_0", "Child": "D1_1_1"},
            {"Parent": "D1_1_1", "Child": "D1_1_2"},
            # {"Parent": "D1_2_0", "Child": "D1_2_1"},
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
        },
    }
    opsInfo["ParameterJson"] = {
        "D1_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7" ,
                "services": {
                    "python39-cpu": {
                        "image": "ubuntu:20.04",
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
                "docker exec -it python39-cpu pip install pip==22.3.1",
                "docker exec -it python39-cpu pip install setuptools==65.6.3",
                # 基本套件 SSH套件 -------------------------------------------
                "docker exec -it python39-cpu pip install gitpython==3.1.27",
                "docker exec -it python39-cpu pip install python-dotenv==0.21.0",
                "docker exec -it python39-cpu pip install paramiko==2.11.0",
                # 資料處理套件 -----------------------------------------------
                "docker exec -it python39-cpu pip install pandas==1.4.4",
                "docker exec -it python39-cpu pip install numpy==1.22.4",
                "docker exec -it python39-cpu pip install scipy==1.8.1",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install pyodbc==4.0.34",
                "docker exec -it python39-cpu pip install SQLAlchemy==1.4.41",
                "docker exec -it python39-cpu pip install psycopg2-binary==2.9.3",
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install matplotlib==3.6.0",  # 繪圖套件
                "docker exec -it python39-cpu pip install seaborn==0.12.1",  # 繪圖套件
                "docker exec -it python39-cpu pip install pillow==9.3.0",  # 圖片處理套件
                "docker exec -it python39-cpu pip install Flask==2.2.2",  # 網頁套件
                "docker exec -it python39-cpu pip install streamlit==1.16.0",  # 網頁套件
                "docker exec -it python39-cpu pip install tqdm==4.64.1",  # 進度條套件
            ],
        },
        "D1_1_2": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install impyla==0.18.0",  # 用於讀取Hive資料
                "docker exec -it python39-cpu pip install hdfs==2.7.0",  # 用於讀取HDFS資料
                "docker exec -it python39-cpu pip install pyspark==3.3.1",  # pyspark套件
                "docker exec -it python39-cpu pip install py4j==0.10.9.7",  # pyspark相依套件
                "docker exec -it python39-cpu pip install fastparquet==0.8.3",  # 讀取parquet檔案
                # 機器學習套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install scikit-learn==1.1.3",  # 機器學習套件
                "docker exec -it python39-cpu pip install scikit-surprise==1.1.3",  # 機器學習套件
                "docker exec -it python39-cpu pip install mlxtend==0.21.0",  # 機器學習套件
                "docker exec -it python39-cpu pip install networkx==2.8.8",  # 網路分析
                "docker exec -it python39-cpu pip install xgboost==1.6.2",  # 梯度提升樹
                "docker exec -it python39-cpu pip install catboost==1.0.6",  # 梯度提升樹
                "docker exec -it python39-cpu pip install lightgbm==3.3.1",  # 梯度提升樹
                "docker exec -it python39-cpu pip install m2cgen==0.10.0",  # 模型轉換套件
                "docker exec -it python39-cpu pip install evidently==0.2.0",  # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install nltk==3.8",  # 自然語言處理套件
                "docker exec -it python39-cpu pip install rake_nltk==1.0.6",  # 自然語言處理套件
                "docker exec -it python39-cpu pip install gensim==4.2.0",  # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install torch==1.13.0",  # 深度學習套件
                "docker exec -it python39-cpu pip install torchvision==0.14.0",  # 深度學習套件
                "docker exec -it python39-cpu pip install torchaudio==0.13.0",  # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install tensorflow==2.11.0",
                # 自動機器學習套件 -------------------------------------------------
                "docker exec -it python39-cpu pip install pycaret==3.0.0rc2",  # 自動機器學習套件
                "docker exec -it python39-cpu pip install autokeras==1.0.20",  # 自動機器學習套件
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
                        "image": "nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04",
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
                "docker exec -it python39-gpu pip install pip==22.3.1",
                "docker exec -it python39-gpu pip install setuptools==65.6.3",
                # 基本套件 SSH套件 -------------------------------------------
                "docker exec -it python39-gpu pip install gitpython==3.1.27",
                "docker exec -it python39-gpu pip install python-dotenv==0.21.0",
                "docker exec -it python39-gpu pip install paramiko==2.11.0",
                # 資料處理套件 -----------------------------------------------
                "docker exec -it python39-gpu pip install pandas==1.4.4",
                "docker exec -it python39-gpu pip install numpy==1.22.4",
                "docker exec -it python39-gpu pip install scipy==1.8.1",
                # 資料庫套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install pyodbc==4.0.34",
                "docker exec -it python39-gpu pip install SQLAlchemy==1.4.41",
                "docker exec -it python39-gpu pip install psycopg2-binary==2.9.3",
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install matplotlib==3.6.0",  # 繪圖套件
                "docker exec -it python39-gpu pip install seaborn==0.12.1",  # 繪圖套件
                "docker exec -it python39-gpu pip install pillow==9.3.0",  # 圖片處理套件
                "docker exec -it python39-gpu pip install Flask==2.2.2",  # 網頁套件
                "docker exec -it python39-gpu pip install streamlit==1.16.0",  # 網頁套件
                "docker exec -it python39-gpu pip install tqdm==4.64.1",  # 進度條套件
            ],
        },
        "D1_2_2": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                # Hadoop套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install impyla==0.18.0",  # 用於讀取Hive資料
                "docker exec -it python39-gpu pip install hdfs==2.7.0",  # 用於讀取HDFS資料
                "docker exec -it python39-gpu pip install pyspark==3.3.1",  # pyspark套件
                "docker exec -it python39-gpu pip install py4j==0.10.9.7",  # pyspark相依套件
                "docker exec -it python39-gpu pip install fastparquet==0.8.3",  # 讀取parquet檔案
                # 機器學習套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install scikit-learn==1.1.3",  # 機器學習套件
                "docker exec -it python39-gpu pip install scikit-surprise==1.1.3",  # 機器學習套件
                "docker exec -it python39-gpu pip install mlxtend==0.21.0",  # 機器學習套件
                "docker exec -it python39-gpu pip install networkx==2.8.8",  # 網路分析
                "docker exec -it python39-gpu pip install xgboost==1.6.2",  # 梯度提升樹
                "docker exec -it python39-gpu pip install catboost==1.0.6",  # 梯度提升樹
                "docker exec -it python39-gpu pip install lightgbm==3.3.1",  # 梯度提升樹
                "docker exec -it python39-gpu pip install m2cgen==0.10.0",  # 模型轉換套件
                "docker exec -it python39-gpu pip install evidently==0.2.0",  # 模型評估套件
                # 自然語言套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install nltk==3.8",  # 自然語言處理套件
                "docker exec -it python39-gpu pip install rake_nltk==1.0.6",  # 自然語言處理套件
                "docker exec -it python39-gpu pip install gensim==4.2.0",  # 自然語言處理套件
                # Pytorch套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cu116",   # 深度學習套件
                "docker exec -it python39-gpu pip install torchvision==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu116",   # 深度學習套件
                "docker exec -it python39-gpu pip install torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116",   # 深度學習套件
                # Tensorflow 套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install tensorflow-gpu==2.12.0 ",
                # 自動機器學習套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install pycaret==3.0.0rc2",  # 自動機器學習套件
                "docker exec -it python39-gpu pip install autokeras==1.0.20",  # 自動機器學習套件
                # 其他套件 -------------------------------------------------
                "docker exec -it python39-gpu pip install gym==0.26.2",  # 遊戲場套件
            ],
        },
        "D1_6_0": {
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
                        , "cleanvolumes": [
                            "/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data"
                            # 本機位置 : 遠端位置
                        ]
                        , "ports": [
                            "5432:5432"
                        ]
                    }
                }
            }
        },
        "D2_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-cpu": {
                        "image": "vicying/python:3.9.13-cpu-0.1.4",
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
        "D2_2_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-gpu": {
                        "image": "vicying/python:3.9.13-gpu-0.1.4",
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
        },
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

