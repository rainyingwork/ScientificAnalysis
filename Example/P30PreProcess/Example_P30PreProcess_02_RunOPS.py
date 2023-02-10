import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]
    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P30PreProcess"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["ParameterJson"] = {
            "R0_0_1": {
                "FunctionType": "GetXYData",
                "DataTime" : dateInfo['DataTime'],
                "MakeDataKeys": ["common_001"],
                "MakeDataInfo": [
                    {"DataType": "Y", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [1]},
                    {"DataType": "X", "Product": "Example", "Project": "P29RawData", "Version": "R0_0_1", "DTDiff": 0, "ColumnNumbers": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]},
                ],
            },
            "P0_0_1": {
                "FunctionType": "PPTagText",
                "DataTime": dateInfo['DataTime'],
                "DataVersion" : "R0_0_1",
            }
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

