import os

class FuncResultFunction():

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        if functionVersionInfo['FunctionType'] == "FRGetFuncResult":
            resultDict, globalObjectDict = self.frGetFuncResult(functionVersionInfo)
        return resultDict, globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def frGetFuncResult(self, fvInfo):
        functionRestlt, globalObjectDict = self.getFuncResult(fvInfo)
        return functionRestlt, globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def getFuncResult(self, fvInfo):
        import pickle
        from dotenv import load_dotenv
        from package.common.common.osbasic.SSHCtrl import SSHCtrl
        repProduct = fvInfo["RepProduct"]
        repProject = fvInfo["RepProject"]
        repVersion = fvInfo["RepOPSVersion"]
        repRecordId = fvInfo["RepOPSRecordId"]
        repFunction = fvInfo["RepFunction"]

        product = fvInfo["Product"]
        project = fvInfo["Project"]
        version = fvInfo["OPSVersion"]
        recordId = fvInfo["OPSRecordId"]
        function = fvInfo["Function"]

        isDownloadRestltObject = fvInfo["IsDownloadRestltObject"] if "IsDownloadRestltObject" in fvInfo.keys() else True
        isDownloadGlobalObject = fvInfo["IsDownloadGlobalObject"] if "IsDownloadGlobalObject" in fvInfo.keys() else True

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl_Storage = SSHCtrl(host=os.getenv("SSH_IP"), port=int(os.getenv("SSH_PORT")), user=os.getenv("SSH_USER"),passwd=os.getenv("SSH_PASSWD"))

        repFunctionRestlt ,repGlobalObjectDict = {} , {}

        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, version, str(recordId),function)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(repProduct, repProject, repVersion, str(repRecordId), repFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        if isDownloadRestltObject == True:
            sshCtrl_Storage.downloadFile("/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"), exeFunctionRDir, "FunctionRestlt.pickle"),"{}/{}".format(exeFunctionLDir, "RepFunctionRestlt.pickle"))
            with open('{}/{}'.format(exeFunctionLDir, 'RepFunctionRestlt.pickle'), 'rb') as fr:
                repFunctionRestlt = pickle.load(fr)
        if isDownloadGlobalObject == True:
            sshCtrl_Storage.downloadFile("/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"), exeFunctionRDir, "GlobalObjectDict.pickle"),"{}/{}".format(exeFunctionLDir, "RepGlobalObjectDict.pickle"))
            with open('{}/{}'.format(exeFunctionLDir, 'RepGlobalObjectDict.pickle'), 'rb') as god:
                repGlobalObjectDict = pickle.load(god)
        del sshCtrl_Storage
        return repFunctionRestlt, repGlobalObjectDict