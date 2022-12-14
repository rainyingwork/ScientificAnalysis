import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["creatdecops"]
            , "Product": ["Example"]
            , "Project": ["P33PycaretUse"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["ParameterJson"] = {
            "R0_0_1": {
                "FunctionType": "GetXYData"
                , "DataTime": dateInfo['DataTime']
                , "MakeMaxColumnCount": 30
                , "MakeDataKeys": ["common_001"]
                , "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0,
                     "ColumnNumbers": [1]}
                    , {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0,
                       "ColumnNumbers": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]}
                ]
            }
            , "P0_0_1": {
                "FunctionType": "PPTagText"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "R0_0_1"
            }
            , "M0_0_1": {
                "FunctionType": "TagFilter"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "P0_0_1"
                , "ModelFunction": "Lasso"
                , "ModelParameter": {
                    "TopK": 10
                    , "Filter": 0.000000001
                }
            }
            , "R0_0_2": {
                "FunctionType": "GetXYDataByFunctionRusult"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "M0_0_1"
            }
            , "P0_0_2": {
                "FunctionType": "PPTagText"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "R0_0_2"
            }
            , "M0_0_2": {
                "FunctionType": "AutoML"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "P0_0_2"
                , "ModelFunction": "TrainPycaretDefult"
                , "ModelParameter": {
                    "TaskType": "Classification"
                }
            }
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

