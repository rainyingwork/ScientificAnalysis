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
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": ['M0_0_1']
        , "OrdFunctionArr": [
        ]
        , "FunctionMemo": {}
    }
    opsInfo["ParameterJson"] = {}
    executeOPSCommon.main(opsInfo)
