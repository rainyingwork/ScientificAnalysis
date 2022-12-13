import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon_local as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["UnitTest"]
        , "Project": ["PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_6_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": ["M0_6_0"]
        # "ExeFunctionArr": ["R0_0_1","P0_0_1","M0_0_1","UP0_0_1"]
        # , "RepOPSRecordId": 9999
        # , "RepFunctionArr": ["R0_0_1","P0_0_1","M0_0_1"]
        # , "RunFunctionArr": ["UP0_0_1"]
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {
            "M0_6_0" :""
        }
    }
    opsInfo["ParameterJson"] = {
        "M0_6_0": {}
    }
    executeOPSCommon.main(opsInfo)
