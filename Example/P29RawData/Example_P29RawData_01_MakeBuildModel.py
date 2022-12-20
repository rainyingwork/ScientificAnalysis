import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"],
        "Product": ["Example"],
        "Project": ["P29RawData"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "R0_0_0"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_0", "Child": "R0_0_1"},
        ],
        "FunctionMemo": {
            "R0_0_0": "Juice資料文本製作",
            "R0_0_1": "Juice資料塞入分析資料庫",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

