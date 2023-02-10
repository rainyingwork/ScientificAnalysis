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
            "Project": ["P35TensorFlow"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        # opsInfo["OPSRecordId"] = [9999]
        opsInfo["OPSOrderJson"] =  {
            "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","UP0_0_1"],
            # "RepOPSRecordId": 9999,
            # "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1"],
            # "RunFunctionArr": ["UP0_0_1"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_1", "Child": "P0_0_1"},
                {"Parent": "P0_0_1", "Child": "M0_0_1"},
                {"Parent": "M0_0_1", "Child": "UP0_0_1"},
            ],
            "FunctionMemo": {
                "R0_0_1" :"撈取相關資料",
                "P0_0_1" :"預處理相關資料",
                "M0_0_1" :"TF模型訓練",
                "UP0_0_1":"TF模型驗證",
            },
        }
        opsInfo["ParameterJson"] = {
            "R0_0_1": {
                "FunctionType": "GetXYData",
                "DataTime" : dateInfo['DataTime'],
                "MakeMaxColumnCount": 30,
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                ]
            },
            "P0_0_1": {
                "FunctionType": "PPTagText",
                "DataTime": dateInfo['DataTime'],
                "DataVersion" : "R0_0_1",
            },
            "M0_0_1": {
                "FunctionType": "",
                "DataVersion": "P0_0_1",
            },
            "UP0_0_1": {
                "FunctionType": "",
                "DataVersion": "M0_0_1",
            },
        }
    executeOPSCommon.main(opsInfo)
