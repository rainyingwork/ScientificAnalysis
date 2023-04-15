import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P81DataPerception"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr":["S0_0_1","R0_0_0","R0_0_1"],
        "OrdFunctionArr":[
            {"Parent":"S0_0_1","Child":"R0_0_0"},
            {"Parent":"S0_0_1","Child":"R0_0_1"}
        ],
        "FunctionMemo":{
            "S0_0_1" : "Juice資料塞入正規資料庫",
            "R0_0_0": "Juice資料文本製作",
            "R0_0_1": "Juice資料塞入分析資料庫",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)