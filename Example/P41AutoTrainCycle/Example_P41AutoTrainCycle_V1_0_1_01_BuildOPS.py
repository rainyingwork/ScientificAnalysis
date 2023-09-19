import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P41AutoTrainCycle"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["S0_0_1"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "S0_0_1": "製作資料，製作相關的登入資料",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

