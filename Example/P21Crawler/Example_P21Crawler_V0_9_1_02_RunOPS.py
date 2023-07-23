import os , copy
import OPSCommon as executeOPSCommon
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2017-01-01"},{"DataTime": "2018-01-01"}
    ]

    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P21Crawler"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_9_1"]
        # opsInfo["OPSRecordId"] = []
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["C0_9_1"],
            # "RepOPSRecordId": 0,
            # "RepFunctionArr": [""],
            # "RunFunctionArr": [""],
            "OrdFunctionArr": [
            ],
            "FunctionMemo": {
                "C0_9_1": "爬取每日上市，根據日期撈取當日所有上市價格資料，TWLSPriceByDay",
            },
        }
        opsInfo["ParameterJson"] = {
            "C0_9_1": {
                "FunctionType": "FreeFunction",
                "DataTime": dateInfo["DataTime"],
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)




