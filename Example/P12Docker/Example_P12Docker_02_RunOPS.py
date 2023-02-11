import os , copy
from dotenv import load_dotenv
import OPSCommon as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P12Docker"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["D1_1_0", "D1_1_1"],
        "OrdFunctionArr": [
            {"Parent": "D1_1_0", "Child": "D1_1_1"},
        ],
        "FunctionMemo": {
            "D1_1_0": "部屬Ubuntu",
            "D1_1_1": "更新Ubuntu",
        },
    }
    opsInfo["ParameterJson"] = {
        "D1_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "ubuntu": {
                        "image": "ubuntu:20.04",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/Docker/Ubuntu/Volumes/Library:/Library",
                            "/Docker/Ubuntu/Volumes/Data:/Data",
                        ],
                    },
                },
            },
        },
        "D1_1_1": {
            "FunctionType": "RunDockerCmdStr",
            "DockerCmdStrs": [
                "docker exec -it ubuntu apt update",
            ],
        },
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)






