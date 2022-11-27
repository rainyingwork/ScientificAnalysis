import os ,sys ,copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import common.P01_OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["Example"]
        , "Project": ["P31TagFilter"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
        ]
        , "FunctionMemo": {
            "R0_0_1": "撈取相關資料"
            , "P0_0_1": "處理相關資料"
            , "M0_0_1": "參數過濾"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
