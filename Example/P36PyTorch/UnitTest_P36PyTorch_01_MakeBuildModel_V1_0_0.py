import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["Example"]
        , "Project": ["P36PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R1_0_1", "P1_0_1", "M1_0_1", "UP1_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R1_0_1", "Child": "P1_0_1"}
            , {"Parent": "P1_0_1", "Child": "M1_0_1"}
            , {"Parent": "M1_0_1", "Child": "UP1_0_1"}
        ]
        , "FunctionMemo": {
            "V1_0_0" : "使用PyTorch辨識圖片是屬於什麼類別"
            , "R1_0_1": "撈取相關圖形資料"
            , "P1_0_1": "預處理相關圖形資料"
            , "M1_0_1": "神經網路使用PyTorch"
            , "UP1_0_1": "使用模型預測結果"
        }
    }
    opsInfo["ParameterJson"] = {
        "R1_0_1": {}
        , "P1_0_1": {
            "FunctionType": ""
            , "DataVersion": "R1_0_1"
        }
        , "M1_0_1": {
            "FunctionType": ""
            , "DataVersion": "P1_0_1"
        }
        , "UP1_0_1": {
            "FunctionType": ""
            , "DataVersion": "P1_0_1"
            , "ItemNo": 30
        }
    }
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)