import os , copy
import OPSCommonLocal as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P23Standard"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_0_1"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "S0_0_1": "Juice資料塞入正規資料庫",
        },
    }
    opsInfo["ParameterJson"] = {
        "S0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
