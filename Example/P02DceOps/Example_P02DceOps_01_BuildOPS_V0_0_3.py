import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P02DceOps"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_3"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_1_1", "R0_1_2", "R0_1_3", "P0_1_1", "P0_1_2", "P0_1_3", "M0_1_1"],
        "OrdFunctionArr": [
            {"Parent": "R0_1_1", "Child": "P0_1_1"},
            {"Parent": "R0_1_2", "Child": "P0_1_1"},
            {"Parent": "R0_1_3", "Child": "P0_1_1"},
            {"Parent": "P0_1_1", "Child": "P0_1_3"},
            {"Parent": "P0_1_2", "Child": "P0_1_3"},
            {"Parent": "P0_1_3", "Child": "M0_1_1"},
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

