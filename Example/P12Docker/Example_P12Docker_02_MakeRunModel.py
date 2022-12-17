import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
from dotenv import load_dotenv
import OPSCommon_local as executeOPSCommon

load_dotenv(dotenv_path="env/postgresql.env")

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["Example"]
        , "Project": ["P12Docker"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D0_0_1"]
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {
            "D0_0_1": ""
        }
    }

    opsInfo["ParameterJson"] = {
        "D0_0_1" :{
            "FunctionType": "RunContainerByDockerComposeInfo"
            , "DockerComposeInfo" : {
                "version": "3.7"
                , "services" : {
                    "postgresql" : {
                        "image":"postgres:14.2-alpine"
                        , "restart": "always"
                        , "environment" : {
                            "POSTGRES_DB" : "postgres"
                            , "POSTGRES_USER" : os.getenv("POSTGRES_USERNAME")
                            , "POSTGRES_PASSWORD" : os.getenv("POSTGRES_PASSWORD")
                            , "PGDATA" : "/lib/postgresql/data"
                        }
                        , "volumes" : [
                            "/mfs/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data"
                            # 本機位置 : 遠端位置
                        ]
                        , "ports" : [
                            "5432:5432"
                        ]
                    }
                }
            }
        }
    }
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)






