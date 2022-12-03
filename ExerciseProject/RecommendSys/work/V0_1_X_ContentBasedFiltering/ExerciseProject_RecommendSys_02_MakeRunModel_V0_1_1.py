import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_1_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_1_1","S0_1_2"]
        # , "RepOPSRecordId": 1149
        # , "RepFunctionArr": ["R0_1_1","P0_1_1"]
        # , "RunFunctionArr": ["S0_1_2"]
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {
            "S0_1_1": "電影主要資料匯入"
            , "S0_1_2": "電影細項資料匯入"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
