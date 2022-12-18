import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon_local as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["ScientificAnalysisSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["SystemSet1_0_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["SYS1_0_1","SYS1_0_2"]
        , "OrdFunctionArr": [
            {"Parent": "SYS1_0_1", "Child": "SYS1_0_2"}
        ]
        , "FunctionMemo": {
            "SYS1_0_1": "安裝基本套件"
            , "SYS1_0_2": "安裝Docker"
        }
    }
    opsInfo["ParameterJson"] = {
        "SYS1_0_1": {
            "FunctionType": "CmdStrRun"
            , "CmdStrs" : [
                "apt install -y zip "
                , "apt install -y unzip "
            ]
        }
        ,"SYS1_0_2": {
            "FunctionType": "CmdStrRun"
            , "CmdStrs" : [
                "sudo apt-get install -y docker.io"
            ]
        }
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

