import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["CustomerRetention"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","CR0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "CR0_0_1"}
        ]
        , "FunctionMemo": {
            "R0_0_1" :"撈取相關資料"
            , "P0_0_1" :"處理相關資料"
            , "CR0_0_1" : "產生相關圖表"
        }
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {}
        , "P0_0_1": {}
        , "CR0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)