import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon


if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P36Pycaret"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","R0_0_2","P0_0_2","M0_0_2"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"},
            {"Parent": "P0_0_1", "Child": "M0_0_1"},
            {"Parent": "M0_0_1", "Child": "R0_0_2"},
            {"Parent": "R0_0_2", "Child": "P0_0_2"},
            {"Parent": "P0_0_2", "Child": "M0_0_2"},
        ],
        "FunctionMemo": {
            "R0_0_1": "撈取資料，撈取XYData資料",
            "P0_0_1": "處理資料，預處理XYData資料",
            "M0_0_1": "模型訓練，使用Lasso做參數過濾",
            "R0_0_2": "撈取資料，撈取Lasso的XYData資料，使用M0_0_1的結果",
            "P0_0_2": "處理資料，預處理Lasso的XYData資料",
            "M0_0_2": "模型訓練，使用AutoML做模型訓練",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
