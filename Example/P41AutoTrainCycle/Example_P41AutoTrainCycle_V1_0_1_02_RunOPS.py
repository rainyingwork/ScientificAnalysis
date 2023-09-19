import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2017-05-01"},{"DataTime": "2017-05-02"},{"DataTime": "2017-05-03"},{"DataTime": "2017-05-04"},{"DataTime": "2017-05-05"},
        {"DataTime": "2017-05-06"},{"DataTime": "2017-05-07"},{"DataTime": "2017-05-08"},{"DataTime": "2017-05-09"},{"DataTime": "2017-05-10"},
        {"DataTime": "2017-05-11"},{"DataTime": "2017-05-12"},{"DataTime": "2017-05-13"},{"DataTime": "2017-05-14"},{"DataTime": "2017-05-15"},
        {"DataTime": "2017-05-16"},{"DataTime": "2017-05-17"},{"DataTime": "2017-05-18"},{"DataTime": "2017-05-19"},{"DataTime": "2017-05-20"},
        {"DataTime": "2017-05-21"},{"DataTime": "2017-05-22"},{"DataTime": "2017-05-23"},{"DataTime": "2017-05-24"},{"DataTime": "2017-05-25"},
        {"DataTime": "2017-05-26"},{"DataTime": "2017-05-27"},{"DataTime": "2017-05-28"},{"DataTime": "2017-05-29"},{"DataTime": "2017-05-30"},{"DataTime": "2017-05-31"},
        {"DataTime": "2017-06-01"},{"DataTime": "2017-06-02"},{"DataTime": "2017-06-03"},{"DataTime": "2017-06-04"},{"DataTime": "2017-06-05"},
        {"DataTime": "2017-06-06"},{"DataTime": "2017-06-07"},{"DataTime": "2017-06-08"},{"DataTime": "2017-06-09"},{"DataTime": "2017-06-10"},
        {"DataTime": "2017-06-11"},{"DataTime": "2017-06-12"},{"DataTime": "2017-06-13"},{"DataTime": "2017-06-14"},{"DataTime": "2017-06-15"},
        {"DataTime": "2017-06-16"},{"DataTime": "2017-06-17"},{"DataTime": "2017-06-18"},{"DataTime": "2017-06-19"},{"DataTime": "2017-06-20"},
        {"DataTime": "2017-06-21"},{"DataTime": "2017-06-22"},{"DataTime": "2017-06-23"},{"DataTime": "2017-06-24"},{"DataTime": "2017-06-25"},
        {"DataTime": "2017-06-26"},{"DataTime": "2017-06-27"},{"DataTime": "2017-06-28"},{"DataTime": "2017-06-29"},{"DataTime": "2017-06-30"},
    ]
    for dateInfo in dateInfoArr:
        print(dateInfo['DataTime'])
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P41AutoTrainCycle"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V1_0_1"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr":  ["S0_0_1", ],
            "OrdFunctionArr": [
            ],
            "FunctionMemo": {
                "S0_0_1": "製作資料，製作相關的登入資料",
            },
        }
        opsInfo["ParameterJson"] = {
            "S0_0_1" :{
                "DataTime": dateInfo['DataTime'],
            }
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)

