import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P41AutoTrainCycle"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V2_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["RBD1_1_0","RBD1_1_1"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "RBD1_1_0": "製作資料，製作相關的登入標籤資料",
            "RBD1_1_1": "製作資料，製作相關的登入模型資料",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

