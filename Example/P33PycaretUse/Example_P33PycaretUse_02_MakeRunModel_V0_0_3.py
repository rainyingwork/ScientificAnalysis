import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["runops"]
            , "Product": ["Example"]
            , "Project": ["P33PycaretUse"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["ParameterJson"] = {
            "R0_0_3": {
                "FunctionType": "GetXYDataByDatabaseRusult"
                , "DataTime" : dateInfo['DataTime']
                , "DatabaseProduct": "Example"
                , "DatabaseProject": "P33PycaretUse"
                , "DatabaseOPSVersion": "V0_0_1"
                , "DatabaseOPSRecord": 640
                , "DatabaseFunction": "M0_0_1"
            }
            , "P0_0_2": {
                "FunctionType": "PPTagText"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "R0_0_3"
            }
            , "M0_0_3": {
                "FunctionType": "AutoML"
                , "DataTime": dateInfo['DataTime']
                , "DataVersion": "P0_0_2"
                , "ModelFunction": "UsePycaretModelByDatabaseRusult"
                , "DatabaseProduct": "Example"
                , "DatabaseProject": "P33PycaretUse"
                , "DatabaseOPSVersion": "V0_0_1"
                , "DatabaseOPSRecord": 863
                , "DatabaseFunction": "M0_0_2"
                , "DatabaseModelName": "CatBoostClassifier"
            }
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)
