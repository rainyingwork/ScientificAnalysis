import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P12Docker"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0", "D1_1_1"],
        "OrdFunctionArr": [
            {"Parent": "D1_1_0", "Child": "D1_1_1"},
        ],
        "FunctionMemo": {
            "D1_1_0": "部屬Ubuntu",
            "D1_1_1": "更新Ubuntu",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)