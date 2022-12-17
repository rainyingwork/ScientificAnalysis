class SSHCmdFunction():

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["DockerComposeInfo"] :
                resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "CmdStrRun":
            otherInfo = self.scCmdStrRun(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def scCmdStrRun(self, fvInfo):
        otherInfo = {}
        otherInfo = self.runCmdStr(fvInfo, otherInfo)
        return otherInfo

    # ================================================= CommonFunction =================================================


    # ==================================================   PPTagText  ==================================================

    @classmethod
    def runCmdStr(self, fvInfo, otherInfo):
        import os
        from dotenv import load_dotenv
        from package.common.osbasic.SSHCtrl import SSHCtrl
        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )
        cmdStrArr = fvInfo["CmdStrs"]
        for cmdStr in cmdStrArr:
            sshCtrl.execSSHCommand(cmdStr)
        del sshCtrl
        return otherInfo
