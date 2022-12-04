import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_2_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_2_1","S0_2_2"]
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {
            "S0_2_1": "電影基本資料"
            , "S0_2_2": "用戶評分資料"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)