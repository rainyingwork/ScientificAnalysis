import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon_local as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["Example"]
        , "Project": ["P36PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": [
            'M0_0_1','M0_0_2','M0_0_3','M0_0_4','M0_0_5',
            'M0_0_6','M0_0_7','M0_0_8','M0_0_9',
            'M0_0_10Train','M0_0_10Test',
            'M0_0_11Train','M0_0_11Test',
            'M0_0_12','M0_0_13','M0_0_14','M0_0_15',
            'M0_0_16','M0_0_17','M0_0_18']
        , "OrdFunctionArr": [
            {"Parent": "M0_0_10Train", "Child": "M0_0_10Test"}
            , {"Parent": "M0_0_11Train", "Child": "M0_0_11Test"}
        ]
        , "FunctionMemo": {}
    }
    opsInfo["ParameterJson"] = {}
    executeOPSCommon.main(opsInfo)
