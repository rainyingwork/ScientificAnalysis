import os , copy
import OPSCommonLocal as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {

    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0","D1_2_0","D1_3_0","D1_3_1","D1_3_2","D1_4_0","D1_4_1","D1_4_2","D2_3_0","D2_4_0"],
        "RepOPSRecordId": 9999,
        "RepFunctionArr": [],
        "RunFunctionArr": ["D1_1_0"],
        "OrdFunctionArr": [
            # {"Parent": "D1_3_0", "Child": "D1_3_1"},
            # {"Parent": "D1_3_1", "Child": "D1_3_2"},
            # {"Parent": "D1_4_0", "Child": "D1_4_1"},
            # {"Parent": "D1_4_1", "Child": "D1_4_2"},
        ],
        "FunctionMemo": {
            "D1_1_0": "部署Python39-CPU基底",
            # "D1_1_0": "部署PostgreSQL",
            # "D1_2_0": "部署MongoDB",
            # "D1_3_0": "部署Python39基底",
            # "D1_3_1": "安裝Python39基本套件",
            # "D1_3_2": "安裝Python39機器學習套件",
            # "D1_4_0": "部署Python39-GPU基底",
            # "D1_4_1": "安裝Python39-GPU基本套件",
            # "D1_4_2": "安裝Python39-GPU機器學習套件",
            # "D2_3_0": "部署完整Python39-CPU",
            # "D2_4_0": "部署完整Python39-GPU",
        },
    }
    opsInfo["ParameterJson"] = {
        "D1_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7" ,
                "services": {
                    "python39-cpu": {
                        "image": "ubuntu:20.04",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y" ,
                        },
                        "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library",
                            "/Docker/Python39/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)

import os , copy
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["BuildOPS"],
        "Product": ["common"],
        "Project": ["SciAnaSys"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0", "D1_1_1", "D1_1_2", "D1_2_0", "D1_2_1", "D1_2_2", "D1_6_0", "D2_1_0", "D2_2_0"],
        "OrdFunctionArr": [
            {"Parent": "D1_1_0", "Child": "D1_1_1"},
            {"Parent": "D1_1_1", "Child": "D1_1_2"},
            {"Parent": "D1_2_0", "Child": "D1_2_1"},
            {"Parent": "D1_2_1", "Child": "D1_2_2"},
        ],
        "FunctionMemo": {
            "D1_1_0": "部署Python39-CPU基底",
            "D1_1_1": "安裝Python39-CPU基本套件",
            "D1_1_2": "安裝Python39-CPU機器學習套件",
            "D1_2_0": "部署Python39-GPU基底",
            "D1_2_1": "安裝Python39-GPU基本套件",
            "D1_2_2": "安裝Python39-GPU機器學習套件",
            "D1_6_0": "部署PostgreSQL",
            "D2_1_0": "部署完整Python39-CPU",
            "D2_2_0": "部署完整Python39-GPU",
        },
    }
    opsInfo["ParameterJson"] = {}
    opsInfo["ResultJson"] = {}

    executeOPSCommon.main(opsInfo)