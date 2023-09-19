import os , copy
import OPSCommonLocal as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P51OLAP"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = ["9999"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["O0_0_1", "O0_0_2", "O0_0_3"],
        "RepOPSRecordId": 9999,
        "RepFunctionArr": ["O0_0_1", "O0_0_2",],
        "RunFunctionArr": ["O0_0_3"],
        "OrdFunctionArr": [
            {"Parent": "O0_0_1", "Child": "O0_0_3"},
            {"Parent": "O0_0_2", "Child": "O0_0_3"},
        ],
        "FunctionMemo": {
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
