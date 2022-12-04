import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_1_2"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_1_1", "R0_1_2", "P0_1_1", "M0_1_1", "M0_1_2","UP0_1_1"]
        , "RepOPSRecordId": 1214
        # # , "RepFunctionArr": ["R0_1_1", "R0_1_2"]
        # # , "RunFunctionArr": ["P0_1_1"]
        , "RepFunctionArr": ["R0_1_1", "R0_1_2", "P0_1_1", "M0_1_1", "M0_1_2"]
        , "RunFunctionArr": ["UP0_1_1"]
        , "OrdFunctionArr": [
             {"Parent": "R0_1_1", "Child": "P0_1_1"}
            , {"Parent": "R0_1_2", "Child": "P0_1_1"}
            , {"Parent": "P0_1_1", "Child": "M0_1_1"}
            , {"Parent": "M0_1_1", "Child": "M0_1_2"}
            , {"Parent": "M0_1_2", "Child": "UP0_1_1"}
        ]
        , "FunctionMemo": {
            "R0_1_1": "撈取電影主要資料"
            , "R0_1_2": "撈取電影細項資料"
            , "P0_1_1": "資料處理"
            , "M0_1_1": "模型使用-關鍵字提取模型rake_nltk"
            , "M0_1_2": "模型使用-詞袋模型CountVectorizer"
            , "UP0_1_1": "使用產品-電影推薦"
        }
        , "referenceURL": "https://ithelp.ithome.com.tw/articles/10219033"
    }
    opsInfo["ParameterJson"] = {
        "R0_1_1": {"FunctionType": "GetSQLData"}
        , "R0_1_2": {"FunctionType": "GetSQLData"}
        , "P0_1_1": {}
        , "M0_1_1": {}
        , "M0_1_2": {}
        , "UP0_1_1": {"MovieName":"Toy Story"}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
