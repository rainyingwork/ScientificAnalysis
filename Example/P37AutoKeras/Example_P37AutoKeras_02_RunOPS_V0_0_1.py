import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P37AutoKeras"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2","M0_0_2"],
            # "RepOPSRecordId": 2607,
            # "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2",],
            # "RunFunctionArr": ["M0_0_2"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_1", "Child": "P0_0_1"},
                {"Parent": "P0_0_1", "Child": "M0_0_1"},
                {"Parent": "M0_0_1", "Child": "R0_0_2"},
                {"Parent": "R0_0_2", "Child": "P0_0_2"},
                {"Parent": "P0_0_2", "Child": "M0_0_2"},
            ],
            "FunctionMemo": {
                "R0_0_1": "撈取XYData資料",
                "P0_0_1": "預處理XYData資料",
                "M0_0_1": "使用Lasso做參數過濾",
                "R0_0_2": "撈取Lasso的XYData資料，使用M0_0_1的結果",
                "P0_0_2": "預處理Lasso的XYData資料",
                "M0_0_2": "使用AutoDL做模型訓練",
            },
        }
        opsInfo["ParameterJson"] = {
            "R0_0_1": {
                "FunctionType": "GetXYData",
                "DataTime" : dateInfo['DataTime'],
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Filter", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "max({})", "ColumnNumbers": [3], "HavingSQL":["> 0"]},
                    {"DataType": "Y", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "sum({})", "ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "sum({})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "avg({})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "max({})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "min({})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "count({})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0 ,"GFuncSQL": "count(distinct {})", "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                ]
            },
            "P0_0_1": {
                "FunctionType": "PPTagText",
                "DataTime": dateInfo['DataTime'],
                "DataVersion" : "R0_0_1",
            },
            "M0_0_1": {
                "FunctionType":"TagFilter",
                "DataTime": dateInfo['DataTime'],
                "DataVersion": "P0_0_1",
                "ModelFunction":"Lasso",
                "ModelParameter":{
                    "TopK": 20,
                    "Filter":0.000000001,
                },
            },
            "R0_0_2": {
                "FunctionType": "GetXYDataByFunctionRusult",
                "DataTime" : dateInfo['DataTime'],
                "DataVersion": "M0_0_1",
            },
            "P0_0_2": {
                "FunctionType": "PPTagText",
                "DataTime": dateInfo['DataTime'],
                "DataVersion": "R0_0_2",
            },
            "M0_0_2": {
                "FunctionType": "AutoDL",
                "DataTime": dateInfo['DataTime'],
                "DataVersion": "P0_0_2",
                "ModelFunction": "TrainAutoKerasDefult",
                "ModelParameter": {
                    "TaskType": "Classification",
                    "MaxTrials": 20,
                    "TrainEpochs": 30,
                },
            },
        }

        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

