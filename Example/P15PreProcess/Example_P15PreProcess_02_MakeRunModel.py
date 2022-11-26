import os ,sys ,copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import pandas
import common.P01_OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["runops"]
            , "Product": ["Example"]
            , "Project": ["P15PreProcess"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExecuteArr": ["R0_0_1","P0_0_1"]
            , "OrderArr": [
                {"Parent": "R0_0_1", "Child": "P0_0_1"}
            ]
            , "FunctionMemo": {
                "R0_0_1": "撈取相關資料"
                , "P0_0_1": "處理相關資料"
            }
        }
        opsInfo["ParameterJson"] = {
            "R0_0_1": {
                "FunctionType": "GetXYData"
                , "DataTime" : dateInfo['DataTime']
                , "MakeDataKeys": ["common_001"]
                , "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P14RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [1]}
                    , {"DataType": "X", "Product": "Example", "Project": "P14RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
                ]
            }
            , "P0_0_1": {
                "FunctionType": "PPTagText"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion" : "R0_0_1"
            }
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

