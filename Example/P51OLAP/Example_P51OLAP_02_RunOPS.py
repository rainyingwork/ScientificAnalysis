import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P51OLAP"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSRecordId"] = []
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_3", "UP0_0_1"],
        # "RepOPSRecordId": 2701,
        # "RepFunctionArr": ["R0_0_1"],
        # "RunFunctionArr": ["UP0_0_1"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_3", "Child": "UP0_0_1"},
        ],
        "FunctionMemo": {
            "R0_0_1": "",
            "UP0_0_1": "",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
