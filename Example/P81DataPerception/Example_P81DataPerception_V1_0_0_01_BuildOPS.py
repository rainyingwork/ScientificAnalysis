import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P81DataPerception"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr":["DP0_0_1"],
        "OrdFunctionArr":[
        ],
        "FunctionMemo":{
        },
    }
    opsInfo["ParameterJson"] = {
    }
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)