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
            , "Project": ["P36Pycaret"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_3", "P0_0_2", "M0_0_3"],
            "RepOPSRecordId": 9,
            "RepFunctionArr": ["R0_0_3", "P0_0_2"],
            "RunFunctionArr": ["M0_0_3"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_3", "Child": "P0_0_2"},
                {"Parent": "P0_0_2", "Child": "M0_0_3"},
            ],
            "FunctionMemo": {
                "R0_0_3": "撈取相關資料",
                "P0_0_2": "處理相關資料",
                "M0_0_3": "使用相關模型",
            },
        }
        opsInfo["ParameterJson"] = {
            "R0_0_3": {
                "FunctionType": "GetXYDataByDatabaseRusult",
                "DataTime" : dateInfo['DataTime'],
                "DatabaseProduct": "Example",
                "DatabaseProject": "P36Pycaret",
                "DatabaseOPSVersion": "V0_0_1",
                "DatabaseOPSRecord": 6,
                "DatabaseFunction": "M0_0_1",
            },
            "P0_0_2": {
                "FunctionType": "PPTagText",
                "DataTime": dateInfo['DataTime'],
                "DataVersion": "R0_0_3",
            },
            "M0_0_3": {
                "FunctionType": "AutoML",
                "DataTime": dateInfo['DataTime'],
                "DataVersion": "P0_0_2",
                "ModelFunction": "UsePycaretModelByDatabaseRusult",
                "DatabaseProduct": "Example",
                "DatabaseProject": "P36Pycaret",
                "DatabaseOPSVersion": "V0_0_1",
                "DatabaseOPSRecord": 6,
                "DatabaseFunction": "M0_0_2",
                "DatabaseModelName": "LogisticRegression",
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)
