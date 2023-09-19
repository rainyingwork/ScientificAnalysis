import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon
import json
import datetime
from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity

if __name__ == "__main__":

    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P41AutoTrainCycle"],
    }
    # ================================================== V3_1_1 ==================================================

    V3_1_1_OPSInfo = copy.deepcopy(basicInfo)
    V3_1_1_OPSInfo["OPSVersion"] = ["V3_1_1"]
    V3_1_1_OPSInfo["OPSOrderJson"] = {
        "ExeFunctionArr": [
            "R0_1_X", "P0_1_X", "M0_1_X",
        ],
        "OrdFunctionArr": [
            {"Parent": "R0_1_X", "Child": "P0_1_X"},
            {"Parent": "P0_1_X", "Child": "M0_1_X"},
        ],
        "FunctionMemo": {
            "R0_1_X": "撈取資料",
            "P0_1_X": "處理資料",
            "M0_1_X": "模型預測",
            "V3_1_0": "塞入文本資料",
            "V3_1_1": "塞入預測資料",
        },
    }
    V3_1_1_OPSInfo["ParameterJson"] = {
        "R0_1_X": {
            "FunctionType": "GetXYData",
            "DataTime": None,
            "MakeDataKeys": ["common_001"],
            "MakeDataInfo": [
                {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -1,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -2,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -3,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -4,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -5,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -6,"ColumnNumbers": [1]},
                {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": -7,"ColumnNumbers": [1]},
            ]
        },
        "P0_1_X": {
            "FunctionType": "PPTagText",
            "DataTime": None,
            "DataVersion": "R0_1_X",
        },
        "M0_1_X": {},
    }
    V3_1_1_OPSInfo["ResultJson"] = {}
    executeOPSCommon.main(V3_1_1_OPSInfo)

    # ================================================== V3_1_2 ==================================================

    V3_1_2_OPSInfo = copy.deepcopy(basicInfo)
    V3_1_2_OPSInfo["OPSVersion"] = ["V3_1_2"]
    V3_1_2_OPSInfo["OPSOrderJson"] = {
        "ExeFunctionArr": [
            "R0_11_X", "P0_11_X", "M0_11_X",
            "R0_12_X", "P0_12_X", "M0_12_X"
        ],
        "OrdFunctionArr": [
            {"Parent": "R0_11_X", "Child": "P0_11_X"},
            {"Parent": "P0_11_X", "Child": "M0_11_X"},
            {"Parent": "M0_11_X", "Child": "R0_12_X"},
            {"Parent": "R0_12_X", "Child": "P0_12_X"},
            {"Parent": "P0_12_X", "Child": "M0_12_X"},
        ],
        "FunctionMemo": {
            "R0_11_X": "撈取資料，撈取XYData資料",
            "P0_11_X": "處理資料，預處理XYData資料",
            "M0_11_X": "模型使用，使用Lasso做參數過濾",
            "R0_12_X": "撈取資料，撈取Lasso的XYData資料，使用M0_0_1的結果",
            "P0_12_X": "處理資料，預處理Lasso的XYData資料",
            "M0_12_X": "模型訓練，使用AutoML做模型訓練",
        },
    }
    V3_1_2_OPSInfo["ParameterJson"] = {
        "R0_11_X": {},
        "P0_11_X": {},
        "M0_11_X": {},
        "R0_12_X": {},
        "P0_12_X": {},
        "M0_12_X": {},
    }
    V3_1_2_OPSInfo["ResultJson"] = {}
    executeOPSCommon.main(V3_1_2_OPSInfo)

    # ================================================== V3_1_3 ==================================================

    V3_1_3_OPSInfo = copy.deepcopy(basicInfo)
    V3_1_3_OPSInfo["OPSVersion"] = ["V3_1_3"]
    V3_1_3_OPSInfo["OPSOrderJson"] = {
        "ExeFunctionArr": [
            "R0_20_X", "P0_20_X",
            "R0_21_X",
            "R0_22_X",
            "R0_23_X",
            "R0_24_X",
            "R0_25_X",
            "R0_26_X",
            "R0_27_X", "P0_29_X",
            "M0_21_X",
            "M0_22_X",
            "M0_23_X",
            "M0_24_X",
        ],
        "OrdFunctionArr": [
            {"Parent": "R0_20_X", "Child": "P0_20_X"},
            {"Parent": "P0_20_X", "Child": "R0_21_X"},
            {"Parent": "P0_20_X", "Child": "R0_22_X"},
            {"Parent": "P0_20_X", "Child": "R0_23_X"},
            {"Parent": "P0_20_X", "Child": "R0_24_X"},
            {"Parent": "P0_20_X", "Child": "R0_25_X"},
            {"Parent": "P0_20_X", "Child": "R0_26_X"},
            {"Parent": "P0_20_X", "Child": "R0_27_X"},
            {"Parent": "R0_21_X", "Child": "P0_29_X"},
            {"Parent": "R0_22_X", "Child": "P0_29_X"},
            {"Parent": "R0_23_X", "Child": "P0_29_X"},
            {"Parent": "R0_24_X", "Child": "P0_29_X"},
            {"Parent": "R0_25_X", "Child": "P0_29_X"},
            {"Parent": "R0_26_X", "Child": "P0_29_X"},
            {"Parent": "R0_27_X", "Child": "P0_29_X"},
            {"Parent": "P0_29_X", "Child": "M0_23_X"},
            {"Parent": "M0_21_X", "Child": "M0_23_X"},
            {"Parent": "M0_22_X", "Child": "M0_23_X"},
            {"Parent": "M0_23_X", "Child": "M0_24_X"},
        ],
        "FunctionMemo": {
            "R0_20_X": "撈取資料",
            "P0_20_X": "處理資料",
            "R0_21_X": "拆分資料",
            "R0_22_X": "拆分資料",
            "R0_23_X": "拆分資料",
            "R0_24_X": "拆分資料",
            "R0_25_X": "拆分資料",
            "R0_26_X": "拆分資料",
            "R0_27_X": "拆分資料",
            "P0_29_X": "合併資料",
            "M0_21_X": "撈取模型",
            "M0_22_X": "撈取模型",
            "M0_23_X": "驗證模型",
            "M0_24_X": "選擇模型",
        },
    }
    V3_1_3_OPSInfo["ParameterJson"] = {
        "R0_20_X": {},
        "P0_20_X": {},
        "R0_21_X": {},
        "R0_22_X": {},
        "R0_23_X": {},
        "R0_24_X": {},
        "R0_25_X": {},
        "R0_26_X": {},
        "R0_27_X": {},
        "P0_29_X": {},
        "M0_21_X": {},
        "M0_22_X": {},
        "M0_23_X": {},
        "M0_24_X": {},
    }
    V3_1_3_OPSInfo["ResultJson"] = {}
    executeOPSCommon.main(V3_1_3_OPSInfo)

    # ================================================== V3_1_4 ==================================================

    V3_1_4_OPSInfo = copy.deepcopy(basicInfo)
    V3_1_4_OPSInfo["OPSVersion"] = ["V3_1_4"]
    V3_1_4_OPSInfo["OPSOrderJson"] = {}
    V3_1_4_OPSInfo["ParameterJson"] = {}
    V3_1_4_OPSInfo["ResultJson"] = {}
    executeOPSCommon.main(V3_1_4_OPSInfo)
