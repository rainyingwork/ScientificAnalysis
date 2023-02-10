import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"]
            , "Product": ["Example"]
            , "Project": ["P36Pycaret"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_3", "P0_0_2", "M0_0_3"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_3", "Child": "P0_0_2"},
                {"Parent": "P0_0_2", "Child": "M0_0_3"},
            ],
            "FunctionMemo": {
                "R0_0_3": "撈取資料庫XYData資料，使用其他OPSRecord的結果",
                "P0_0_2": "預處理資料庫XYData資料",
                "M0_0_2": "使用模型預測，使用其他OPSRecord的模型",
            },
        }
        opsInfo["ParameterJson"] = {
            "R0_0_3": {
                "FunctionType": "GetXYDataByDatabaseRusult",
                "DataTime" : dateInfo['DataTime'],
                "DatabaseProduct": "Example",
                "DatabaseProject": "P36Pycaret",
                "DatabaseOPSVersion": "V0_0_1",
                "DatabaseOPSRecord": 151,
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
                "DatabaseOPSVersion": "V0_0_2",
                "DatabaseOPSRecord": 155,
                "DatabaseFunction": "M0_0_2",
                "DatabaseModelName": "LogisticRegression",
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)
