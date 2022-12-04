import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_2_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_2_1", "R0_2_2", "P0_2_1", "M0_2_1" ,"UP0_2_1","UP0_2_2"]
        # , "RepOPSRecordId": 000
        # , "RepFunctionArr": [""]
        # , "RunFunctionArr": [""]
        , "OrdFunctionArr": [
            {"Parent": "R0_2_1", "Child": "P0_2_1"}
            , {"Parent": "R0_2_2", "Child": "P0_2_1"}
            , {"Parent": "P0_2_1", "Child": "M0_2_1"}
            , {"Parent": "M0_2_1", "Child": "UP0_2_1"}
            , {"Parent": "M0_2_1", "Child": "UP0_2_2"}
        ]
        , "FunctionMemo": {
            "R0_2_1": "撈取電影主要資料"
            , "R0_2_2": "撈取用戶評分資料"
            , "P0_2_1": "資料處理"
            , "M0_2_1": "USER-ITEM Matrix"
            , "UP0_2_1": "ITEM-ITEM Similarity Use"
            , "UP0_2_2": "USER-USER Similarity Use"
        }
        , "referenceURL": "https://ithelp.ithome.com.tw/articles/10219511"
    }
    opsInfo["ParameterJson"] = {
        "R0_2_1": {"FunctionType": "GetSQLData"}
        , "R0_2_2": {"FunctionType": "GetSQLData"}
        , "P0_2_1": {}
        , "M0_2_1": {}
        , "UP0_2_1": {"MovieName": "Bad Boys (1995)"}
        , "UP0_2_2": {"UserID": "10"}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
