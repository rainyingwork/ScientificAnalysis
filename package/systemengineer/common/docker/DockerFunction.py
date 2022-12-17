import os
import time , datetime
import pprint
import math, pandas
import json
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class DockerFunction(CommonFunction):

    def __init__(self):
        from package.common.osbasic.SSHCtrl import SSHCtrl
        load_dotenv(dotenv_path="env/ssh.env")
        self.sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["DockerComposeInfo"] :
                resultDict[key] = functionVersionInfo[key]

        if functionVersionInfo['FunctionType'] == "RunContainerByDockerComposeInfo":
            otherInfo = self.dRunContainerByDockerComposeInfo(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def dRunContainerByDockerComposeInfo(self, fvInfo):
        otherInfo = {}
        otherInfo = self.runContainerByDockerComposeInfo(fvInfo, otherInfo)
        return otherInfo

    # ================================================= CommonFunction =================================================


    # ==================================================   PPTagText  ==================================================

    @classmethod
    def runContainerByDockerComposeInfo(self, fvInfo, otherInfo):
        servicesInfo = fvInfo["DockerComposeInfo"]["services"]
        commandArr = []
        for key in servicesInfo.keys():
            containerName = key
            serviceInfo = servicesInfo[key]
            self.sshCtrl.execSSHCommand("docker stop {}".format(containerName))
            self.sshCtrl.execSSHCommand("docker rm {}".format(containerName))
            for volumeStr in serviceInfo["volumes"]:
                storagePath = volumeStr.split(":")[0]
                self.sshCtrl.execSSHCommand("rm -rf {}".format(storagePath))
                self.sshCtrl.execSSHCommand("mkdir {}".format(storagePath))
                self.sshCtrl.execSSHCommand("chmod -R 777 {}".format(storagePath))
            commandArr.append("--name {}".format(key))
            for infoKey in serviceInfo.keys():
                if type(serviceInfo[infoKey]) == type(''):
                    commandArr.append("--{} {}".format(infoKey, serviceInfo[infoKey]))
                elif type(serviceInfo[infoKey]) == type([]):
                    for values in serviceInfo[infoKey]:
                        commandArr.append("--{} {}".format(infoKey, values))
                elif type(serviceInfo[infoKey]) == type({}):
                    for parameterName in serviceInfo[infoKey].keys():
                        commandArr.append("--{} {}={}".format(infoKey, parameterName, serviceInfo[infoKey][parameterName]))
            commandStr = "docker run -itd \n    {} ".format('\n    '.join(commandArr))
            self.sshCtrl.execSSHCommand(commandStr)
        return otherInfo
