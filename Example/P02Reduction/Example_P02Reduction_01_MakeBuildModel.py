import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["Example"]
        , "Project": ["P02Reduction"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    # opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSOrderJson"] = {
    #     "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2","M0_0_2"]
    #     , "OrderArr": [
    #         {"Parent": "R0_0_1", "Child": "P0_0_1"}
    #         , {"Parent": "P0_0_1", "Child": "M0_0_1"}
    #         , {"Parent": "M0_0_1", "Child": "R0_0_2"}
    #         , {"Parent": "R0_0_2", "Child": "P0_0_2"}
    #         , {"Parent": "P0_0_2", "Child": "M0_0_2"}
    #     ]
    #     , "FunctionMemo": {
    #         "R0_0_1": "撈取相關資料"
    #         , "P0_0_1": "處理相關資料"
    #         , "M0_0_1": "參數過濾"
    #     }
    # }
    # opsInfo["ParameterJson"] = {}
    # opsInfo["ResultJson"] = {}
    # executeOPSCommon.main(opsInfo)

    opsInfo["OPSVersion"] = ["V0_0_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "P0_0_1", "M0_0_1", "R0_0_2", "P0_0_2", "M0_0_2"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
            , {"Parent": "M0_0_1", "Child": "R0_0_2"}
            , {"Parent": "R0_0_2", "Child": "P0_0_2"}
            , {"Parent": "P0_0_2", "Child": "M0_0_2"}
        ]
        , "FunctionMemo": {
            "R0_0_1": "撈取相關資料"
            , "P0_0_1": "處理相關資料"
            , "M0_0_1": "參數過濾"
            , "R0_0_2": "根據參數過濾撈取相關資料"
            , "P0_0_2": "處理相關資料"
            , "M0_0_2": "訓練模型"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

