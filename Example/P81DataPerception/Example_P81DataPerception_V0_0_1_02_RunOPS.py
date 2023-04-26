import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2023-01-01"},{"DataTime": "2023-01-02"},{"DataTime": "2023-01-03"},{"DataTime": "2023-01-04"},{"DataTime": "2023-01-05"},
        {"DataTime": "2023-01-06"},{"DataTime": "2023-01-07"},{"DataTime": "2023-01-08"},{"DataTime": "2023-01-09"},{"DataTime": "2023-01-10"},
        {"DataTime": "2023-01-11"},{"DataTime": "2023-01-12"},{"DataTime": "2023-01-15"}, # 故意缺13、14號
        {"DataTime": "2023-01-16"},{"DataTime": "2023-01-17"},{"DataTime": "2023-01-18"},{"DataTime": "2023-01-19"},{"DataTime": "2023-01-20"},
        # {"DataTime": "2023-01-21"},{"DataTime": "2023-01-22"},{"DataTime": "2023-01-23"},{"DataTime": "2023-01-24"},{"DataTime": "2023-01-25"},
        # {"DataTime": "2023-01-26"},{"DataTime": "2023-01-27"},{"DataTime": "2023-01-28"},{"DataTime": "2023-01-29"},{"DataTime": "2023-01-30"},
        # {"DataTime": "2023-01-31"},
    ]
    for dateInfo in dateInfoArr :
        print(dateInfo["DataTime"])
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P81DataPerception"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": [
                "S0_0_1", "S0_0_11", "S0_0_12",
                "R0_0_0", "R0_0_1"
            ],
            "OrdFunctionArr": [
                {"Parent": "S0_0_1", "Child": "R0_0_0"},
                {"Parent": "S0_0_1", "Child": "R0_0_1"},
            ],
            "FunctionMemo": {
                "S0_0_1": "Juice資料塞入正規資料庫",
                "S0_0_11": "Juice資料Purchase等於CH塞入正規資料庫",
                "S0_0_12": "Juice資料Purchase等於MM塞入正規資料庫",
                "R0_0_0": "Juice資料文本製作",
                "R0_0_1": "Juice資料塞入分析資料庫",
            },
        }
        opsInfo["ParameterJson"] = {
            "S0_0_1": {
                "DataTime": dateInfo["DataTime"],
            },
            "S0_0_11": {
                "DataTime": dateInfo["DataTime"],
            },
            "S0_0_12": {
                "DataTime": dateInfo["DataTime"],
            },
            "R0_0_0": {
                "FunctionType": "MakeTagText",
                "Product": "Example",
                "Project": "P81DataPerception",
                "Version": "R0_0_0",
                "DataTime": dateInfo["DataTime"],
                "FeatureType": "General",
                "FilePath": "Example/P81DataPerception/file/TagText/TagR0_0_0.json",
            },
            "R0_0_1": {
                "FunctionType": "ExeSQLStrs",
                "DataTime": dateInfo["DataTime"],
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)




