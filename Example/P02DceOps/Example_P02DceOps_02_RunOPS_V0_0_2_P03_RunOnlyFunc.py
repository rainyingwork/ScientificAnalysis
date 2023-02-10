import os , copy
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOnlyFunc"],
            "Product": ["Example"],
            "Project": ["P02DceOps"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_2"]
        opsInfo["OPSRecordId"] = [53]
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


