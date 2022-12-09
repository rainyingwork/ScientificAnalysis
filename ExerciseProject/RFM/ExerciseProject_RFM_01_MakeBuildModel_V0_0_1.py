import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RFM"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_0_1"]
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {
            "S0_0_1": "資料塞入標準資料庫"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)