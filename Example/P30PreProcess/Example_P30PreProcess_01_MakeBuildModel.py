import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"],
        "Product": ["Example"],
        "Project": ["P30PreProcess"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "P0_0_1"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"},
        ],
        "FunctionMemo": {
            "R0_0_1": "撈取相關資料",
            "P0_0_1": "處理相關資料",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
