import os , copy
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"] ,
        "Product": ["Example"] ,
        "Project": ["P01Basic"] ,
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSRecordId"] = []
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["C0_0_1", "O0_0_1", "S0_0_1", "R0_0_1", "P0_0_1", "M0_0_1"] ,
        # "RepOPSRecordId": 0 ,
        # "RepFunctionArr": [""] ,
        # "RunFunctionArr": [""] ,
        "OrdFunctionArr": [
            {"Parent": "C0_0_1", "Child": "O0_0_1"} ,
            {"Parent": "O0_0_1", "Child": "S0_0_1"} ,
            {"Parent": "S0_0_1", "Child": "R0_0_1"} ,
            {"Parent": "R0_0_1", "Child": "P0_0_1"} ,
            {"Parent": "P0_0_1", "Child": "M0_0_1"} ,
        ],
        "FunctionMemo": {
            "C0_0_1": "",
            "O0_0_1": "",
            "S0_0_1": "",
            "R0_0_1": "",
            "P0_0_1": "",
            "M0_0_1": "",
        } ,
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
