import os , copy
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["runops"],
            "Product": ["Example"],
            "Project": ["P02DECOPS"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2","M0_0_2"],
            "OrdFunctionArr": [
            ],
            "FunctionMemo": {
                "R0_0_1": "撈取相關資料"
            },
        }
        opsInfo["ParameterJson"] = {}
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

