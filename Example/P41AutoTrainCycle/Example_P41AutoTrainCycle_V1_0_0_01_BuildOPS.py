import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P41AutoTrainCycle"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["O0_0_1", "O0_0_2", "O0_0_3"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "O0_0_1": "塞入資料，將Customers資料塞入",
            "O0_0_2": "塞入資料，將Pings資料塞入",
            "O0_0_3": "塞入資料，將Test資料塞入",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

