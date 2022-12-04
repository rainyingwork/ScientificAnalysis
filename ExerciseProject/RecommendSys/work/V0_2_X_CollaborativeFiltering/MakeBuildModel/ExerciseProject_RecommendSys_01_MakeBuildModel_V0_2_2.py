import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_2_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_2_1", "R0_2_2", "P0_2_1", "M0_2_1", "M0_2_2", "M0_2_3", "UP0_2_1"]
        # , "RepOPSRecordId": 000
        # , "RepFunctionArr": [""]
        # , "RunFunctionArr": [""]
        , "OrdFunctionArr": [
            {"Parent": "T0_0_0", "Child": "T0_0_1"}
            , {"Parent": "R0_2_1", "Child": "P0_2_1"}
            , {"Parent": "R0_2_2", "Child": "P0_2_1"}
            , {"Parent": "P0_2_1", "Child": "M0_2_1"}
            , {"Parent": "M0_2_1", "Child": "M0_2_2"}
            , {"Parent": "M0_2_1", "Child": "M0_2_3"}
            , {"Parent": "M0_2_2", "Child": "UP0_2_1"}
            , {"Parent": "M0_2_3", "Child": "UP0_2_1"}
        ]
        , "FunctionMemo": {
            "R0_2_1": "撈取電影主要資料"
            , "R0_2_2": "撈取用戶評分資料"
            , "P0_2_1": "資料處理"
            , "M0_2_1": "USER-ITEM Matrix"
            , "M0_2_2": "ITEM-ITEM Similarity"
            , "M0_2_3": "USER-USER Similarity"
            , "UP0_2_1": "USER-USER Similarity"
        }
        , "referenceURL": "https://ithelp.ithome.com.tw/articles/10219511"
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)