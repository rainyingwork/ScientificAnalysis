import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P02DceOps"],
        }
        OPSInfo_V0_0_1_1 = copy.deepcopy(basicInfo)
        OPSInfo_V0_0_1_1["OPSVersion"] = ["V0_0_1"]
        OPSInfo_V0_0_1_1["OPSOrderJson"] = {
            "ExeFunctionArr": ["R0_0_1","M0_0_2"],
            "OrdFunctionArr": [
                {"Parent": "R0_0_1", "Child": "M0_0_2"},
            ],
            "FunctionMemo": {
                "R0_0_1": "測試方法",
                "M0_0_2": "測試方法",
            },
        }
        OPSInfo_V0_0_1_1["ParameterJson"] = {}
        OPSInfo_V0_0_1_1["ResultJson"] = {}
        OPSInfo_V0_0_1_1 = executeOPSCommon.main(OPSInfo_V0_0_1_1)
        repOPSRecordId = int(OPSInfo_V0_0_1_1["OPSRecordId"]) # 因為資料庫給的是int64無法進入Json，所以這邊要轉回來
        loopCount = OPSInfo_V0_0_1_1["ResultJson"]['M0_0_2']['LoopCount']
        for _ in range(0,loopCount):
            OPSInfo_V0_0_1_2 = copy.deepcopy(basicInfo)
            OPSInfo_V0_0_1_2["OPSVersion"] = ["V0_0_1"]
            OPSInfo_V0_0_1_2["OPSOrderJson"] = {
                "ExeFunctionArr": ["M0_0_2","M0_0_3"],
                "RepOPSRecordId": int(repOPSRecordId),
                "RepFunctionArr": ["M0_0_2"],
                "RunFunctionArr": ["M0_0_3"],
                "OrdFunctionArr": [
                    {"Parent": "M0_0_2", "Child": "M0_0_3"},
                ],
                "FunctionMemo": {
                    "M0_0_3": "測試方法",
                },
            }
            OPSInfo_V0_0_1_2["ParameterJson"] = {}
            OPSInfo_V0_0_1_2["ResultJson"] = {}
            OPSInfo_V0_0_1_2 = executeOPSCommon.main(OPSInfo_V0_0_1_2)
            repOPSRecordId = int(OPSInfo_V0_0_1_2["OPSRecordId"]) # 因為資料庫給的是int64無法進入Json，所以這邊要轉回來
        OPSInfo_V0_0_1_3 = copy.deepcopy(basicInfo)
        OPSInfo_V0_0_1_3["OPSVersion"] = ["V0_0_1"]
        OPSInfo_V0_0_1_3["OPSOrderJson"] = {
            "ExeFunctionArr": ["M0_0_3", "M0_0_4"],
            "RepOPSRecordId": repOPSRecordId,
            "RepFunctionArr": ["M0_0_3"],
            "RunFunctionArr": ["M0_0_4"],
            "OrdFunctionArr": [
                {"Parent": "M0_0_3", "Child": "M0_0_4"},
            ],
            "FunctionMemo": {
                "M0_0_4": "測試方法",
            },
        }
        OPSInfo_V0_0_1_3["ParameterJson"] = {}
        OPSInfo_V0_0_1_3["ResultJson"] = {}
        OPSInfo_V0_0_1_3 = executeOPSCommon.main(OPSInfo_V0_0_1_3)
        import pprint
        pprint.pprint(OPSInfo_V0_0_1_3)
