class DockerFunction():

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["DockerComposeInfo"] :
                resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "RunContainerByDockerComposeInfo":
            otherInfo = self.dkRunContainerByDockerComposeInfo(functionVersionInfo)
        elif functionVersionInfo['FunctionType'] == "RunDockerCmdStr":
            otherInfo = self.dkRunDockerCmdStr(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def dkRunContainerByDockerComposeInfo(self, fvInfo):
        otherInfo = {}
        otherInfo = self.runContainerByDockerComposeInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def dkRunDockerCmdStr(self, fvInfo):
        otherInfo = {}
        otherInfo = self.runDockerCmdStr(fvInfo, otherInfo)
        return otherInfo

    # ================================================= CommonFunction =================================================


    # ==================================================   PPTagText  ==================================================

    @classmethod
    def runContainerByDockerComposeInfo(self, fvInfo, otherInfo):
        import os
        from dotenv import load_dotenv
        from package.common.common.osbasic.SSHCtrl import SSHCtrl
        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )
        servicesInfo = fvInfo["DockerComposeInfo"]["services"]
        commandArr = []
        for key in servicesInfo.keys():
            containerName = key
            serviceInfo = servicesInfo[key]
            sshCtrl.execSSHCommand("docker stop {}".format(containerName))
            sshCtrl.execSSHCommand("docker rm {}".format(containerName))
            if "volumes_clean" in serviceInfo.keys():
                for volumeStr in serviceInfo["volumes_clean"]:
                    storagePath = volumeStr.split(":")[0]
                    sshCtrl.execSSHCommand("rm -rf {}".format(storagePath))
                    sshCtrl.execSSHCommand("mkdir -p {}".format(storagePath))
                    sshCtrl.execSSHCommand("chmod -R 777 {}".format(storagePath))

            dockerRunParameterMap = {
                "image": None # continue
                , "restart": "--restart"
                , "gpus": "--gpus"
                , "environment": "-e"
                , "volumes": "-v"
                , "volumes_clean": "-v"
                , "ports": "-p"
            }
            commandArr.append("--name {}".format(key))
            for infoKey in serviceInfo.keys():
                if dockerRunParameterMap[infoKey] == None :
                    continue
                if type(serviceInfo[infoKey]) == type(''):
                    commandArr.append("{} {}".format(dockerRunParameterMap[infoKey], serviceInfo[infoKey]))
                elif type(serviceInfo[infoKey]) == type([]):
                    for values in serviceInfo[infoKey]:
                        commandArr.append("{} {}".format(dockerRunParameterMap[infoKey], values))
                elif type(serviceInfo[infoKey]) == type({}):
                    for parameterName in serviceInfo[infoKey].keys():
                        commandArr.append("{} {}={}".format(dockerRunParameterMap[infoKey], parameterName, serviceInfo[infoKey][parameterName]))
            commandArr.append("{}".format(serviceInfo["image"]))
            commandStr = "docker run -itd {} ".format(' '.join(commandArr))
            printCommandStr = "docker run -itd \n    {} ".format('\n    '.join(commandArr))
            sshCtrl.execSSHCommand(commandStr)
            print(printCommandStr)
        del sshCtrl
        return otherInfo

    @classmethod
    def runDockerCmdStr(self, fvInfo, otherInfo):
        import os
        from dotenv import load_dotenv
        from package.common.common.osbasic.SSHCtrl import SSHCtrl
        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )
        cmdStrArr = fvInfo["DockerCmdStrs"]
        for cmdStr in cmdStrArr:
            sshCtrl.execSSHCommand(cmdStr)
        del sshCtrl
        return otherInfo
