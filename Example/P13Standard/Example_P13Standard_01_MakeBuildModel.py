import os ,sys ,copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import common.P01_OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["buildops"]
        , "Product": ["Example"]
        , "Project": ["P13Standard"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr":["S0_0_1"]
        , "OrdFunctionArr":[
        ]
        , "FunctionMemo":{
            "S0_0_1" : "Juice資料塞入正規資料庫"
        }
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)