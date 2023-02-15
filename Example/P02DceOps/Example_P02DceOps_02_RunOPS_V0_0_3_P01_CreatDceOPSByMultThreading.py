import os , copy
import OPSCommon as executeOPSCommon
import datetime

if __name__ == "__main__":
    dateInfoArr = []
    nowTime = datetime.datetime.now()
    nowZeroTime = datetime.datetime(nowTime.year, nowTime.month, nowTime.day, 0, 0, 0, 0)
    startDateStr = "2022-10-01"
    endDateStr = "2022-10-31"
    startDateTime = datetime.datetime.strptime(startDateStr, "%Y-%m-%d")
    endDateTime = datetime.datetime.strptime(endDateStr, "%Y-%m-%d")
    makeDatetime = startDateTime
    while makeDatetime <= endDateTime:
        dateInfoArr.append({"DataTime": makeDatetime.strftime("%Y-%m-%d")})
        makeDatetime = makeDatetime + datetime.timedelta(days=1)

    opsRecordIdArr = []
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["CreatDCEOPS"],
            "Product": ["Example"],
            "Project": ["P02DceOps"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["OPSOrderJson"] = {
            "RunType": "RunDCEOPS",
            "BatchNumber": 202211292300,
            "ExeFunctionArr": ["R0_1_1", "R0_1_2", "R0_1_3", "P0_1_1", "P0_1_2", "P0_1_3", "M0_1_1"],
            "OrdFunctionArr": [
                {"Parent": "R0_1_1", "Child": "P0_1_1"},
                {"Parent": "R0_1_2", "Child": "P0_1_1"},
                {"Parent": "R0_1_3", "Child": "P0_1_1"},
                {"Parent": "P0_1_1", "Child": "P0_1_3"},
                {"Parent": "P0_1_2", "Child": "P0_1_3"},
                {"Parent": "P0_1_3", "Child": "M0_1_1"},
            ],
            "FunctionMemo": {
                "R0_1_1": "撈取相關資料",
                "R0_1_2": "撈取相關資料",
                "R0_1_3": "撈取相關資料",
                "P0_1_1": "資料整合處理",
                "P0_1_2": "處理相關資料",
                "P0_1_3": "資料整合處理",
                "M0_1_1": "訓練模型",
            },
        }
        opsInfo["ParameterJson"] = {
            "R0_1_1": {"DataTime": dateInfo['DataTime']},
            "R0_1_2": {"DataTime": dateInfo['DataTime']},
            "R0_1_3": {"DataTime": dateInfo['DataTime']},
            "P0_1_1": {"DataTime": dateInfo['DataTime']},
            "P0_1_2": {"DataTime": dateInfo['DataTime']},
            "P0_1_3": {"DataTime": dateInfo['DataTime']},
            "M0_1_1": {"DataTime": dateInfo['DataTime']},
        }
        opsInfo["ResultJson"] = {}
        returnOpsInfo = executeOPSCommon.main(opsInfo)
        opsRecordIdArr.append(returnOpsInfo["OPSRecordId"])

    print(opsRecordIdArr)

