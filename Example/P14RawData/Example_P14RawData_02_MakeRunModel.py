import os ,sys ,copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import pandas
import common.P01_OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]

    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["runops"]
            , "Product": ["Example"]
            , "Project": ["P14RawData"]
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["ParameterJson"] = {
            "R0_0_0": {
                "FunctionType": "MakeTagText"
                , "Product" : "Example", "Project" : "P14RawData", "Version" : "R0_0_0"
                , "DataTime" : dateInfo['DataTime']
                , "FeatureType" : "General"
                , "FilePath" : "Example/P14RawData/file/TagText/TagR0_0_0.json"
            }
            , "R0_0_1": {
                "FunctionType": "ExeSQLStrs"
                , "DataTime": dateInfo['DataTime']
            }
        }
        opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

