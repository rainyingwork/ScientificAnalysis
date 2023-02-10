import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P36Pycaret"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_3"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_3", "P0_0_2", "M0_0_2"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_3", "Child": "P0_0_2"},
            {"Parent": "P0_0_2", "Child": "M0_0_2"},
        ],
        "FunctionMemo": {
            "R0_0_3": "撈取資料庫XYData資料，使用其他OPSRecord的結果",
            "P0_0_2": "預處理資料庫XYData資料",
            "M0_0_2": "使用AutoML做模型訓練",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)