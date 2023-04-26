import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    dateInfoArr = [
        {"DataTime": "2023-01-01"},{"DataTime": "2023-01-02"},{"DataTime": "2023-01-03"},{"DataTime": "2023-01-04"},{"DataTime": "2023-01-05"},
        {"DataTime": "2023-01-06"},{"DataTime": "2023-01-07"},{"DataTime": "2023-01-08"},{"DataTime": "2023-01-09"},{"DataTime": "2023-01-10"},
        {"DataTime": "2023-01-11"},{"DataTime": "2023-01-12"},{"DataTime": "2023-01-15"}, # 故意缺13、14號
        {"DataTime": "2023-01-16"},{"DataTime": "2023-01-17"},{"DataTime": "2023-01-18"},{"DataTime": "2023-01-19"},{"DataTime": "2023-01-20"},
        # {"DataTime": "2023-01-21"},{"DataTime": "2023-01-22"},{"DataTime": "2023-01-23"},{"DataTime": "2023-01-24"},{"DataTime": "2023-01-25"},
        # {"DataTime": "2023-01-26"},{"DataTime": "2023-01-27"},{"DataTime": "2023-01-28"},{"DataTime": "2023-01-29"},{"DataTime": "2023-01-30"},
    ]

    dateInfoArr = [
        {"DataTime": "2023-01-16"}
    ]
    for dateInfo in dateInfoArr :
        print(dateInfo["DataTime"])
        basicInfo = {
            "RunType": ["RunOPS"],
            "Product": ["Example"],
            "Project": ["P81DataPerception"],
        }
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V1_0_0"]
        opsInfo["OPSOrderJson"] = {
            "ExeFunctionArr": ["DP0_0_1","DP0_1_1","DP0_1_2"],
            "OrdFunctionArr": [
                {"Parent": "DP0_0_1", "Child": "DP0_1_1"},
                {"Parent": "DP0_1_1", "Child": "DP0_1_2"},
            ],
            "FunctionMemo": {
            },
        }
        opsInfo["ParameterJson"] = {
            "DP0_0_1": {
                "FunctionType": "MakeDataPercoption",
                "RealTableName": "observationdata.standarddata",
                "Product": "Example",
                "Project": "P81DataPerception",
                "TableName": ["S0_0_1","S0_0_11","S0_0_12",],
                "DataTime": dateInfo["DataTime"],
                "PercopColumnName": None,
                "PercopColumnValse": False,
                "PercepCycle": "1D",
            },
            "DP0_1_1": {
                "FunctionType": "CompareDataPercoptionByDT",
                "RealTableName": "observationdata.standarddata",
                "Product": "Example",
                "Project": "P81DataPerception",
                "Tablename": "S0_0_1",
                "DataTime": dateInfo["DataTime"],
                "PercopColumnName": None,
                "PercopColumnValse": False,
                "PercepCycle": "1D",
            },
            "DP0_1_2": {
                "FunctionType": "CompareDataPercoptionByTableName",
                "RealTableName": "observationdata.standarddata",
                "Product": "Example",
                "Project": "P81DataPerception",
                "MainTablename": "S0_0_1",
                "CompareTablenames": ['S0_0_11', 'S0_0_12'],
                "DataTime": dateInfo["DataTime"],
                "PercopColumnName": None,
                "PercopColumnValse": False,
                "PercepCycle": "1D",
            },
        }
        opsInfo["ResultJson"] = {}
        executeOPSCommon.main(opsInfo)
