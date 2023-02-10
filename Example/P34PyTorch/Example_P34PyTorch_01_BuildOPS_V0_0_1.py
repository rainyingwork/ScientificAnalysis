import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["Example"],
        "Project": ["P34PyTorch"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {}
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)