import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2022-01-01"}
    ]

    for dateInfo in dateInfoArr :
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P29RawData"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_1"]
        opsInfo["ParameterJson"] = {
            "R0_0_0": {
                "FunctionType": "MakeTagText",
                "Product" : "Example",
                "Project" : "P29RawData",
                "Version" : "R0_0_0",
                "DataTime" : dateInfo['DataTime'],
                "FeatureType" : "General",
                "FilePath" : "Example/P29RawData/file/TagText/TagR0_0_0.json",
            },
            "R0_0_1": {
                "FunctionType": "ExeSQLStrs",
                "DataTime": dateInfo['DataTime'],
            },
        }
        opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

