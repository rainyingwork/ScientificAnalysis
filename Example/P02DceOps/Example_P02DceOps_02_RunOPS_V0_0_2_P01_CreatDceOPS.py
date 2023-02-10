import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["CreatDCEOPS"],
            "Product": ["Example"],
            "Project": ["P02DceOps"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_2"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_1", "P0_0_1", "M0_0_1", "R0_0_2", "P0_0_2", "M0_0_2"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_1", "Child": "P0_0_1"},
                {"Parent": "P0_0_1", "Child": "M0_0_1"},
                {"Parent": "M0_0_1", "Child": "R0_0_2"},
                {"Parent": "R0_0_2", "Child": "P0_0_2"},
                {"Parent": "P0_0_2", "Child": "M0_0_2"},
            ],
            "FunctionMemo": {
                "R0_0_1": "測試方法",
                "P0_0_1": "測試方法",
                "M0_0_1": "測試方法",
                "R0_0_2": "測試方法",
                "P0_0_2": "測試方法",
                "M0_0_2": "測試方法",
            },
        }
        opsInfo["ParameterJson"] = {}
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

