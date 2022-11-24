import os ,sys ,copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import pandas
import common.P01_OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["Example"]
        , "Project": ["P13Standard"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["ParameterJson"] = {
        "S0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
