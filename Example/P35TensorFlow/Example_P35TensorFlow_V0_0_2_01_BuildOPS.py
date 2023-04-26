import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":

    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P35TensorFlow"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","UP0_0_2"],
        # "RepOPSRecordId": 1269,
        # "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1"],
        # "RunFunctionArr": ["M0_0_2"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "P0_0_1"},
            {"Parent": "P0_0_1", "Child": "M0_0_1"},
            {"Parent": "M0_0_1", "Child": "UP0_0_2"},
        ],
        "FunctionMemo": {
            "V0_0_1" :"利用花卉資料來做分類的神經網路訓練的相關驗證",
            "R0_0_1" :"撈取相關資料",
            "P0_0_1" :"預處理相關資料",
            "M0_0_1" :"TF模型訓練",
            "UP0_0_2":"TF模型驗證",
        },
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {},
        "P0_0_1": {},
        "M0_0_1": {},
        "UP0_0_2": {},
    }
    executeOPSCommon.main(opsInfo)
