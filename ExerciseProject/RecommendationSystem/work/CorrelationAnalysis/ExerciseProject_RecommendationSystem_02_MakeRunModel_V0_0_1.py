import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendationSystem"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_0_1","R0_0_1","P0_0_1","M0_0_1"]
        # , "RepOPSRecordId": 1098
        # , "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2"]
        , "RunFunctionArr": ["S0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "S0_0_1", "Child": "R0_0_1"}
            , {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
        ]
        , "FunctionMemo": {
            "S0_0_1": ""
            , "R0_0_1": ""
            , "P0_0_1": ""
            , "M0_0_1": ""
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
