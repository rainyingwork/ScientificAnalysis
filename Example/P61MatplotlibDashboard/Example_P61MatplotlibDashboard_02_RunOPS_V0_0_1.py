import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommonLocal as executeOPSCommon

if __name__ == "__main__":

    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P61MatplotlibDashboard"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["AC0_0_1"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "AC0_0_1": "製作圖表範本資料",
        },
    }
    opsInfo["ParameterJson"] = {
    }

    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

