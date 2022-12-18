import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
from dotenv import load_dotenv
import OPSCommon_local as executeOPSCommon

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
        , "RepFunctionArr": ["D1_1_0","D1_2_0","D1_2_1","D1_2_2"]
        , "RunFunctionArr": ["D1_3_0"]
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

    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

