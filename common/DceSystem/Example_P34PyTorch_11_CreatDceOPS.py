import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["CreatDCEOPS"]
        , "Product": ["Example"]
        , "Project": ["P34PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] =  {
        "RunType": "RunDCEOPS",
        "BatchNumber": "DCESystem",
        "ExeFunctionArr": [
            "P0_0_10","M0_0_10","UP0_0_10",
        ],
        "OrdFunctionArr": [
            {"Parent": "P0_0_10", "Child": "M0_0_10"},{"Parent": "M0_0_10", "Child": "UP0_0_10"},
        ],
        "FunctionMemo": {
            "M0_0_10":"使用CNN進行圖片分類",
        },
    }
    opsInfo["ParameterJson"] = {
        "P0_0_10":{},"M0_0_10":{},"UP0_0_10":{},
    }
    executeOPSCommon.main(opsInfo)
