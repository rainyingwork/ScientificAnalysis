import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon_local as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["UnitTest"]
        , "Project": ["PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    # opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","UP0_0_1"]
        # , "RepOPSRecordId": 9999
        # , "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1"]
        # , "RunFunctionArr": ["UP0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
            , {"Parent": "M0_0_1", "Child": "UP0_0_1"}
        ]
        , "FunctionMemo": {
            "R0_0_1" :"撈取相關圖形資料"
            , "P0_0_1" :"預處理相關圖形資料"
            , "M0_0_1" : "神經網路使用PyTorch"
            , "M0_0_2" : "使用模型預測結果"
        }
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {}
        , "P0_0_1": {
            "FunctionType": ""
            , "DataVersion": "R0_0_1"
        }
        , "M0_0_1": {
            "FunctionType": ""
            , "DataVersion": "P0_0_1"
        }
        , "UP0_0_1": {
            "FunctionType": ""
            , "DataVersion": "P0_0_1"
            , "ItemNo": 30
        }
    }
    executeOPSCommon.main(opsInfo)
