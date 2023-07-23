import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P21Crawler"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_9_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["C0_9_1", "S0_9_1"],
        # "RepOPSRecordId": 0,
        # "RepFunctionArr": [""],
        # "RunFunctionArr": [""],
        "OrdFunctionArr": [
            # {"Parent": "C0_0_1", "Child": "O0_0_1"},
            # {"Parent": "O0_0_1", "Child": "S0_0_1"},
            # {"Parent": "S0_0_1", "Child": "R0_0_1"},
            # {"Parent": "R0_0_1", "Child": "P0_0_1"},
            # {"Parent": "P0_0_1", "Child": "M0_0_1"},
        ],
        "FunctionMemo": {
            "C0_9_1": "爬取年度上市價格，爬取一個編號該年度的資料，PriceByYear",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)