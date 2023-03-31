import os , copy
from dotenv import load_dotenv
import OPSCommonLocal as executeOPSCommon

load_dotenv(dotenv_path="env/postgresql.env")

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["RunOPS"]
        , "Product": ["common"]
        , "Project": ["SciAnaSys"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V1_0_0"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] = {
        # "ExeFunctionArr": ["D2_1_0", "D2_2_0", "D2_3_0", "D2_6_0", "D2_8_0",],
        "ExeFunctionArr": ["D2_8_0",],
        # "RepOPSRecordId": 9999,
        # "RepFunctionArr": [],
        # "RunFunctionArr": ["D1_2_1", "D1_2_2"],
        "OrdFunctionArr": [
        ],
        "FunctionMemo": {
            "D2_1_0": "部署PY-部署並重開Python39-DCE-Basic",
            "D2_2_0": "部署PY-部署並重開Python39-DCE-Master",
            "D2_3_0": "部署PY-部署並重開Python39-DCE-Slave",
            "D2_6_0": "部署PG-部署並重開PostgreSQL",
            "D2_8_0": "部署JE-部署並重開Jenkins",
        },
    }
    opsInfo["ParameterJson"] = {
        "D2_1_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-basic": {
                        "image": "vicying/python:3.9.13-gpu-0.1.4",
                        "restart": "always",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library",
                            "/Docker/Python39/Volumes/Data:/Data",
                            "/etc/localtime:/etc/localtime",
                        ],
                    },
                },
            },
        },
        "D2_2_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-master": {
                        "image": "vicying/python:3.9.13-gpu-0.1.4",
                        "restart": "always",
                        "gpus": "all",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library",
                            "/Docker/Python39/Volumes/Data:/Data",
                            "/etc/localtime:/etc/localtime",
                        ],
                    },
                },
            },
        },
        "D2_3_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "python39-dce-slave": {
                        "image": "vicying/python:3.9.13-gpu-0.1.4",
                        "restart": "always",
                        "gpus": "all",
                        "environment": {
                            "ACCEPT_EULA": "Y",
                        },
                        "volumes": [
                            "/Docker/Python39/Volumes/Library:/Library",
                            "/Docker/Python39/Volumes/Data:/Data",
                            "/etc/localtime:/etc/localtime",
                        ],
                    },
                },
            },
        },
        "D2_6_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "postgresql": {
                        "image": "postgres:15.1",
                        "restart": "always",
                        "environment": {
                            "POSTGRES_DB": "postgres",
                            "POSTGRES_USER": os.getenv("POSTGRES_USERNAME"),
                            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
                            "PGDATA": "/lib/postgresql/data",
                        },
                        "volumes": [
                            "/Docker/PostgreSQL/Volumes/Data:/lib/postgresql/data",
                            "/etc/localtime:/etc/localtime",
                            # 本機位置 : 遠端位置
                        ],
                        "ports": [
                            "5432:5432",
                        ]
                    },
                },
            },
        },
        "D2_8_0": {
            "FunctionType": "RunContainerByDockerComposeInfo",
            "DockerComposeInfo": {
                "version": "3.7",
                "services": {
                    "jenkins": {
                        "image": "jenkins/jenkins:lts-jdk11",
                        "restart": "always",
                        "privileged": "true",  # 要有root權限才可以安裝套件
                        "environment": {
                            "TZ": "Asia/Taipei",
                            "JAVA_OPTS": "-Duser.timezone=Asia/Taipei",
                        },
                        "volumes_clean": [
                            "/mfs/Docker/Jenkins/Volumes/jenkins_home:/var/jenkins_home",
                        ],
                        "volumes": [
                            "/etc/localtime:/etc/localtime",
                            "/var/run/docker.sock:/var/run/docker.sock",
                        ],
                        "ports": [
                            "8080:8080",
                            "50000:50000",
                        ],
                    },
                },
            },
        },
    }
    opsInfo["ResultJson"] = {}
    executeOPSCommon.main(opsInfo)
