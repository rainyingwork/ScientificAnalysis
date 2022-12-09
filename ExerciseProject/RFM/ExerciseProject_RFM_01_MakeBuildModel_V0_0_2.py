import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RFM"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","M0_0_2","M0_0_3","M0_0_11","CR0_0_1"]
        , "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_1"}
            , {"Parent": "P0_0_1", "Child": "M0_0_2"}
            , {"Parent": "P0_0_1", "Child": "M0_0_3"}
            , {"Parent": "M0_0_1", "Child": "M0_0_11"}
            , {"Parent": "M0_0_2", "Child": "M0_0_11"}
            , {"Parent": "M0_0_3", "Child": "M0_0_11"}
            , {"Parent": "M0_0_11", "Child": "CR0_0_1"}
        ]
        , "FunctionMemo": {
            "R0_0_1" :"撈取相關資料"
            , "P0_0_1" :"預處理相關資料"
            , "M0_0_1" : "計算最近購買日期(Recency)"
            , "M0_0_2" : "計算購買頻率(Frequency)"
            , "M0_0_3" : "計算購買金額(Monetary)"
            , "M0_0_11" : "分群處理"
            , "CR0_0_1" : "詳細分析"
        }
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {}
        , "P0_0_1": {}
        , "M0_0_1": {}
        , "M0_0_2": {}
        , "M0_0_3": {}
        , "M0_0_11": {}
        , "CR0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)