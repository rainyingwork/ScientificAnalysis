import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["runfunc"]
            , "Product": ["Example"]
            , "Project": ["P02Reduction"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_2"]
        opsInfo["OPSRecordId"] = [923]
        opsInfo["RunFunctionArr"] = ["R0_0_1"]
        executeOPSCommon.main(opsInfo)

        opsInfo["RunFunctionArr"] = ["P0_0_1"]
        executeOPSCommon.main(opsInfo)

        opsInfo["RunFunctionArr"] = ["M0_0_1"]
        executeOPSCommon.main(opsInfo)

        opsInfo["RunFunctionArr"] = ["R0_0_2"]
        executeOPSCommon.main(opsInfo)

        opsInfo["RunFunctionArr"] = ["P0_0_2"]
        executeOPSCommon.main(opsInfo)

        opsInfo["RunFunctionArr"] = ["M0_0_2"]
        executeOPSCommon.main(opsInfo)


