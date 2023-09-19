import os , copy;os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon
import json
import datetime
from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2017-06-01"},{"DataTime": "2017-06-02"},{"DataTime": "2017-06-03"},{"DataTime": "2017-06-04"},{"DataTime": "2017-06-05"},
        {"DataTime": "2017-06-06"},{"DataTime": "2017-06-07"},{"DataTime": "2017-06-08"},{"DataTime": "2017-06-09"},{"DataTime": "2017-06-10"},
        {"DataTime": "2017-06-11"},{"DataTime": "2017-06-12"},{"DataTime": "2017-06-13"},{"DataTime": "2017-06-14"},{"DataTime": "2017-06-15"},
        {"DataTime": "2017-06-16"},{"DataTime": "2017-06-17"},{"DataTime": "2017-06-18"},{"DataTime": "2017-06-19"},{"DataTime": "2017-06-20"},
        {"DataTime": "2017-06-21"},{"DataTime": "2017-06-22"},{"DataTime": "2017-06-23"},{"DataTime": "2017-06-24"},{"DataTime": "2017-06-25"},
        {"DataTime": "2017-06-26"},{"DataTime": "2017-06-27"},{"DataTime": "2017-06-28"},{"DataTime": "2017-06-29"},{"DataTime": "2017-06-30"},
    ]
    dateInfoArr = [
        {"DataTime": "2017-06-08"}
    ]

    for dateInfo in dateInfoArr :
        print(dateInfo["DataTime"])
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P41AutoTrainCycle"],
        }

        # ================================================== V3_1_1 ==================================================
        opsVersionEntity = OPSVersionEntity()
        opsvEntity = opsVersionEntity.getOPSVersionByProductProjectOPSVersion("Example","P41AutoTrainCycle","V3_1_0")
        V3_1_1_OPSInfo = copy.deepcopy(basicInfo)
        V3_1_1_OPSInfo["OPSVersion"] = ["V3_1_1"]
        V3_1_1_OPSInfo["OPSOrderJson"] = json.loads(opsvEntity["opsorderjson"])
        V3_1_1_OPSInfo["ParameterJson"] = json.loads(opsvEntity["parameterjson"])
        for exeFunction in V3_1_1_OPSInfo["OPSOrderJson"]["ExeFunctionArr"] :
            V3_1_1_OPSInfo["ParameterJson"][exeFunction]["DataTime"] = (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        V3_1_1_OPSInfo["ResultJson"] = {}

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
                "R0_12_X": "撈取資料，撈取Lasso的XYData資料，使用M0_11_X的結果",
                "P0_12_X": "處理資料，預處理Lasso的XYData資料",
                "M0_12_X": "模型訓練，使用AutoML做模型訓練",
            },
        }
        V3_1_2_OPSInfo["ParameterJson"] = {
            "R0_11_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiff",
                # "FunctionDetailType": None,
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
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
            "P0_11_X": {
                "FunctionType": "PPTagText",
                "FunctionItemType": "ByDTDiff",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "R0_11_X",
            },
            "M0_11_X": {
                "FunctionType": "TagFilter",
                "DataTime":(datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_11_X",
                "ModelFunction": "Lasso",
                "ModelParameter": {
                    "TopK": 5,
                    # "Filter": 0.000000001,
                },
            },
            "R0_12_X": {
                "FunctionType": "GetXYDataByFunctionRusult",
                "FunctionItemType": "ByDTDiff",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "M0_11_X",
            },
            "P0_12_X": {
                "FunctionType": "PPTagText",
                "FunctionItemType": "ByDTDiff",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "R0_12_X",
            },
            "M0_12_X": {
                "FunctionType": "AutoML",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_12_X",
                "ModelFunction": "TrainPycaretDefult",
                "ModelParameter": {
                    "TaskType": "Classification",
                    "IncludeModel": ["xgboost", "dt", "lr", "qda", "svm"],
                    "TopModelCount": 3,
                },
            },
        }
        V3_1_2_OPSInfo["ResultJson"] = {}

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
            # "RepOPSRecordId": ,
            # "RepFunctionArr": [],
            # "RunFunctionArr": [],
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
            "R0_20_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDT",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=0)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-2)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-3)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-5)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-6)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-7)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-8)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-9)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-10)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-11)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-12)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-13)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-14)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-15)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-16)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-17)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-18)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-19)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                    {"Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DT": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-20)).strftime("%Y-%m-%d"),"ColumnNumbers": [1]},
                ]
            },
            "P0_20_X":{
                "FunctionType": "PPTagText",
                "FunctionItemType": "ByDT",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "R0_20_X",
            },
            "R0_21_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_22_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-5)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_23_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-6)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_24_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-7)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_25_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-8)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_26_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-9)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_20_X",
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "R0_27_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiffFromDF",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-10)).strftime("%Y-%m-%d"),
                "MakeDataKeys": ["common_001"],
                "DataVersion": "P0_20_X",
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-1,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-2,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-3,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-4,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-5,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-6,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff":-7,"ColumnNumbers": [1]},
                ]
            },
            "P0_29_X": {
                "FunctionType": "DataConcat",
                "DataVersion": [
                    "R0_21_X",
                    "R0_22_X",
                    "R0_23_X",
                    "R0_24_X",
                    "R0_25_X",
                    "R0_26_X",
                    "R0_27_X",
                ]
            },
            "M0_21_X": {
                "FunctionType": "FreeFunction",
                "Product": "Example",
                "Project": "P41AutoTrainCycle",
                "Version": "V3_1_1",
                "Function": "M0_1_X",
            },
            "M0_22_X": {
                "FunctionType": "FreeFunction",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "Product": "Example",
                "Project": "P41AutoTrainCycle",
                "Version": "V3_1_2",
                "Function": "M0_12_X",
            },
            "M0_23_X": {
                "FunctionType": "FreeFunction",
            },
            "M0_24_X": {
                "FunctionType": "FreeFunction",
                "Product" : "Example",
                "Project" : "P41AutoTrainCycle",
                "Version" : "V3_1_1",
                "Function" : "M0_1_X",
            },
        }
        V3_1_3_OPSInfo["ResultJson"] = {}

        # ================================================== V3_1_4 ==================================================

        V3_1_4_OPSInfo = copy.deepcopy(basicInfo)
        V3_1_4_OPSInfo["OPSVersion"] = ["V3_1_4"]
        V3_1_4_OPSInfo["OPSOrderJson"] = {
            "ExeFunctionArr": [
                "R0_31_X", "P0_32_X", "M0_32_X"
            ],
            "OrdFunctionArr": [
                {"Parent": "R0_31_X", "Child": "P0_32_X"},
                {"Parent": "P0_32_X", "Child": "M0_32_X"},
            ],
            "FunctionMemo": {
                "R0_32_X": "撈取資料",
                "P0_32_X": "處理資料",
                "M0_32_X": "",
            },
        }
        V3_1_4_OPSInfo["ParameterJson"] = {
            "R0_31_X": {
                "FunctionType": "GetXYData",
                "FunctionItemType": "ByDTDiff",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-1)).strftime("%Y-%m-%d"),
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R1_1_1", "DTDiff": 0,"ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P41AutoTrainCycle", "Version": "R0_9_1", "DTDiff": -3,"ColumnNumbers": [2]},
                ]
            },
            "P0_32_X": {
                "FunctionType": "PPTagText",
                "FunctionItemType": "ByDTDiff",
                "DataTime": (datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "R0_31_X",
            },
            "M0_32_X": {
                "FunctionType": "TagFilter",
                "DataTime":(datetime.datetime.strptime(dateInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=-4)).strftime("%Y-%m-%d"),
                "DataVersion": "P0_11_X",
                "ModelFunction": "Lasso",
                "ModelParameter": {
                    "TopK": 5,
                },
            },
        }
        V3_1_4_OPSInfo["ResultJson"] = {}

        if dateInfo["DataTime"] >= "2017-06-08":
            executeOPSCommon.main(V3_1_1_OPSInfo)
        if dateInfo["DataTime"] >= "2017-06-07":
            # 7 (訓練當天) 6 (被預測天) 3 (預測當天) 2 (使用資料) 1 (使用資料)
            executeOPSCommon.main(V3_1_2_OPSInfo)
        if dateInfo["DataTime"] >= "2017-06-07":
            executeOPSCommon.main(V3_1_3_OPSInfo)


