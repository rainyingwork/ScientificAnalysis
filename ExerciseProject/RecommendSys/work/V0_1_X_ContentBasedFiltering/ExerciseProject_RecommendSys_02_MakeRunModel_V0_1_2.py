import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "P0_0_1", "M0_0_1"]
        , "RepOPSRecordId": 1149
        , "RepFunctionArr": ["R0_0_1","P0_0_1"]
        , "RunFunctionArr": ["M0_0_1"]
        # , "RunFunctionArr": ["R0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
        ]
        , "FunctionMemo": {
            "R0_0_1": ""
            , "P0_0_1": ""
            , "M0_0_1": ""
        }
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {"FunctionType": "GetSQLData"}
        , "P0_0_1": {}
        , "M0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
