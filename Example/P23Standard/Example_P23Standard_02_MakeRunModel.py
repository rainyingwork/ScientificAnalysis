import os , copy
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["Example"]
        , "Project": ["P23Standard"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["ParameterJson"] = {
        "S0_0_1": {}
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
