import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P51OLAP"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "UP0_0_1"],
        # "RepOPSRecordId": 0,
        # "RepFunctionArr": [""],
        # "RunFunctionArr": [""],
        "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "UP0_0_1"},
        ],
        "FunctionMemo": {
            "R0_0_1": "",
            "UP0_0_1": "",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)