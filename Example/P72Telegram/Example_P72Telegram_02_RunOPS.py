import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P72Telegram"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSRecordId"] = []
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["Chat0_0_1","Chat0_0_2"],
        # "RepOPSRecordId": 0,
        # "RepFunctionArr": [""],
        # "RunFunctionArr": [""],
        "OrdFunctionArr": [
            {"Parent": "Chat0_0_1", "Child": "Chat0_0_2"},
        ],
        "FunctionMemo": {
            "Chat0_0_1": "訊息保存",
            "Chat0_0_2": "訊息回覆",
        },
    }
    opsInfo["ParameterJson"] = {





    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
