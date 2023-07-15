import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2023-01-01"},
    ]
    for dateInfo in dateInfoArr :
        print(dateInfo["DataTime"])
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P86Doc"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": [
                "S0_0_1",
                "R0_0_0", "R0_0_1"
            ],
            "OrdFunctionArr": [
                {"Parent": "S0_0_1", "Child": "R0_0_0"},
                {"Parent": "S0_0_1", "Child": "R0_0_1"},
            ],
            "FunctionMemo": {
                "S0_0_1": "Juice資料塞入正規資料庫",
                "R0_0_0": "Juice資料文本製作",
                "R0_0_1": "Juice資料塞入分析資料庫",
            },
        }
        opsInfo["ParameterJson"] = {
            "S0_0_1": {
                "DataTime": dateInfo["DataTime"],
            },
            "R0_0_0": {
                "FunctionType": "MakeTagText",
                "Product": "Example",
                "Project": "P86Doc",
                "Version": "R0_0_0",
                "DataTime": dateInfo["DataTime"],
                "FeatureType": "General",
                "FilePath": "Example/P86Doc/file/TagText/TagR0_0_0.json",
            },
            "R0_0_1": {
                "FunctionType": "ExeSQLStrs",
                "DataTime": dateInfo["DataTime"],
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)




