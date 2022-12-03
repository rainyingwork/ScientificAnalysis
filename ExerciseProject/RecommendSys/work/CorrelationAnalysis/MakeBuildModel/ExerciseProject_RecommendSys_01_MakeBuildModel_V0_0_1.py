import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "S0_0_1", "Child": "R0_0_1"}
        ]
        , "FunctionMemo": {
            "S0_0_1": "基本資料匯入"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)