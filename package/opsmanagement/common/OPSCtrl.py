import os , copy , shutil
import threading
import time
from queue import Queue
import pickle
from dotenv import load_dotenv
from package.common.common.osbasic.SSHCtrl import SSHCtrl

class OPSCtrl:

    def __init__(self):
        load_dotenv(dotenv_path="env/ssh.env")
        pass

    def executeOPS(self, opsInfo):
        allGlobalObjectDict = {}
        opsInfo["GlobalObject"] = id(allGlobalObjectDict)
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]
        print("Start OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product , project, opsVersion,opsRecordId))
        opsOrderDict = self.makeCompleteOPSOrderDict(opsInfo['OPSOrderJson'])
        for orderFunctionLayer in opsOrderDict["OrderLayerArr"] :
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer :
                if executeFunction in opsOrderDict["RepFunctionArr"] and executeFunction not in opsOrderDict["NoSLFunctionArr"]:
                    thread = threading.Thread(target=self.replyExecuteFunction, args=(executeFunction,opsInfo,opsOrderDict["RepOPSRecordId"] ,threadQueue))
                    thread.daemon = True
                    time.sleep(0.5), thread.start() , time.sleep(0.5)
                    threadList.append(thread)
                if executeFunction in opsOrderDict["RunFunctionArr"]:
                    isSL = True if executeFunction not in opsOrderDict["NoSLFunctionArr"] else False
                    thread = threading.Thread(target=self.runExecuteFunction, args=(executeFunction,opsInfo,threadQueue,isSL))
                    thread.daemon = True
                    time.sleep(0.5), thread.start() , time.sleep(0.5)
                    threadList.append(thread)
            for thread in threadList:
                thread.join()
            for _ in threadList:
                functionDict = threadQueue.get()
                executeFunction = functionDict["ExecuteFunction"]
                opsInfo["ResultJson"][executeFunction] = functionDict["FunctionRestlt"]
                allGlobalObjectDict[executeFunction] = functionDict["GlobalObjectDict"]
        print("End OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product , project, opsVersion,opsRecordId))

    def executeDCE(self, opsInfo):
        product, project , opsVersion , opsRecordId = opsInfo["Product"] , opsInfo["Project"] , opsInfo["OPSVersion"] , opsInfo["OPSRecordId"]
        print("Start DCE , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product, project,opsVersion,opsRecordId))
        opsOrderDict = self.makeCompleteOPSOrderDict(opsInfo['OPSOrderJson'])
        for orderFunctionLayer in opsOrderDict["OrderLayerArr"]:
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer:
                thread = threading.Thread(target=self.dceExecuteFunction,args=(executeFunction, opsInfo,threadQueue))
                thread.daemon = True
                time.sleep(0.5), thread.start() , time.sleep(0.5)
                threadList.append(thread)
            for thread in threadList:
                thread.join()
            for _ in threadList:
                functionDict = threadQueue.get()
                executeFunction = functionDict["ExecuteFunction"]
                opsInfo["ResultJson"][executeFunction] = functionDict["FunctionRestlt"]
        print("End DCE , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product, project,opsVersion,opsRecordId))

    def runExecuteFunction(self,executeFunction, opsInfo, threadQueue,isSL=True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        print("  Start Function , Version is {}  ".format(executeFunction))
        functionRestlt, globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
        if isSL == True:
            functionRestlt, globalObjectDict = self.saveRestltObject(opsInfo, executeFunction, functionRestlt, globalObjectDict)
            functionRestlt, globalObjectDict = self.uploadRestltObject(opsInfo, executeFunction, functionRestlt, globalObjectDict)
        opsDetailId = self.makeExecuteFunctionInfo(opsInfo, executeFunction, functionRestlt,globalObjectDict)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  End Function , Version is {} , OPSDetailID is {} ".format(executeFunction,opsDetailId))

    def replyExecuteFunction(self,executeFunction, opsInfo , repOPSRecordId , threadQueue):
        self.downloadRestltObject(opsInfo, executeFunction, repOPSRecordId, None, None)
        functionRestlt, globalObjectDict = self.loadRestltObject(opsInfo, executeFunction, repOPSRecordId, None, None)
        functionRestlt, globalObjectDict = self.saveRestltObject(opsInfo, executeFunction, functionRestlt,globalObjectDict)
        functionRestlt, globalObjectDict = self.uploadRestltObject(opsInfo, executeFunction, functionRestlt,globalObjectDict)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  Reply Function , Version is {} , ReplyOPSRecordID is {} ".format(executeFunction,str(repOPSRecordId)))

    def dceExecuteFunction(self,executeFunction, opsInfo,threadQueue):
        sshCtrl_DCEByFunc = SSHCtrl(host=os.getenv("SSH_IP"), port=int(os.getenv("SSH_PORT")), user=os.getenv("SSH_USER"),passwd=os.getenv("SSH_PASSWD")
                                    ,timeout=60, printLog=False, isConnectSSH=True, isConnectSFTP=False)
        product, project, opsVersion, opsRecordId = opsInfo["Product"], opsInfo["Project"], opsInfo["OPSVersion"], opsInfo["OPSRecordId"]
        from package.opsmanagement.common.entity.OPSDetailEntity import OPSDetailEntity
        opsDetailEntityCtrl = OPSDetailEntity()
        isHaveOPSDetailEntity = opsDetailEntityCtrl.isHaveOPSDetailEntityByOPSRecordAndExeFunctionAndState(opsRecordId,executeFunction,state="FINISH")
        if isHaveOPSDetailEntity == True :
            print("  Exist DCE Function , Product is {} , Project is {} , Version is {} , OPSRecordID is {} , Function is {}  ".format(product, project,opsVersion,opsRecordId,executeFunction))
        else :
            print("  Start DCE Function , Product is {} , Project is {} , Version is {} , OPSRecordID is {} , Function is {}  ".format(product, project,opsVersion,opsRecordId,executeFunction))
            sshStr = "docker exec -it python39-cpu python3 /Data/ScientificAnalysis/OPSCommon.py --RunType RunOnlyFunc --Product {} --Project {} --OPSVersion {} --OPSRecordId {} --RunFunctionArr {}"
            sshStr = sshStr.format(product, project, opsVersion, opsRecordId, executeFunction)
            sshCtrl_DCEByFunc.execSSHCommandReturn(sshStr)
        self.downloadRestltObject(opsInfo, executeFunction, opsRecordId, None, None,isDownloadRestltObject=True, isDownloadGlobalObject=False)
        functionRestlt , _ = self.loadRestltObject(opsInfo, executeFunction, opsRecordId, None, None,isLoadRestltObject=True, isLoadGlobalObject=False)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
        })
        del sshCtrl_DCEByFunc , opsDetailEntityCtrl
        print("  End DCE Function , Product is {} , Project is {} , Version is {} , OPSRecordID is {} , Function is {} ".format(product, project,opsVersion,opsRecordId,executeFunction))

    # ================================================== CompleteOPSOrderDict ==================================================
    def makeExecuteFunctionInfo(self,opsInfo, executeFunction, functionRestlt, globalObjectDict):
        from package.opsmanagement.common.entity.OPSDetailEntity import OPSDetailEntity
        opsDetailEntityCtrl = OPSDetailEntity()
        functionInfo = {}
        functionInfo["OPSRecordId"] = opsInfo["OPSRecordId"]
        functionInfo["ExeFunction"] = executeFunction
        functionInfo["ParameterJson"] = opsInfo["ParameterJson"][executeFunction] if executeFunction in opsInfo["ParameterJson"].keys() else {}
        functionInfo["ResultJson"] = functionRestlt
        opsDetailEntityCtrl.setEntity(opsDetailEntityCtrl.makeOPSDetailEntityByFunctionInfo(functionInfo))
        opsDetailEntityCtrl.insertEntity()
        return opsDetailEntityCtrl.getEntityId()

    # ================================================== CompleteOPSOrderDict ==================================================

    def makeCompleteOPSOrderDict(self, opsOrderDict):
        opsOrderDict["RepOPSRecordId"] = opsOrderDict["RepOPSRecordId"] if "RepOPSRecordId" in opsOrderDict.keys() else 0
        opsOrderDict["OrderLayerArr"] = self.makeOrderLayerArr(opsOrderDict)
        opsOrderDict['RunFunctionArr'] , opsOrderDict['RepFunctionArr'] , opsOrderDict['NoSLFunctionArr']  = self.makeRunAndRepFunctionArr(opsOrderDict)
        return opsOrderDict

    def makeOrderLayerArr(self, opsOrderDict):
        exeFunctionArr = opsOrderDict['ExeFunctionArr']
        ordFunctionArr = opsOrderDict['OrdFunctionArr']
        orderLayerArr = []
        tempLaveExeFunctionArr = copy.deepcopy(exeFunctionArr)
        tempCullExeFunctionArr = []
        while len(tempLaveExeFunctionArr) != 0:
            orderLayer = []
            tempInitLaveExecuteFunctionArr = copy.deepcopy(tempLaveExeFunctionArr)
            tempInitCullExecuteFunctionArr = copy.deepcopy(tempCullExeFunctionArr)
            for executeFunction in tempInitLaveExecuteFunctionArr:
                isNotCullParent = False
                for orderFunction in ordFunctionArr:
                    if orderFunction['Child'] == executeFunction and orderFunction['Parent'] not in tempInitCullExecuteFunctionArr:isNotCullParent = True
                if isNotCullParent == False:
                    orderLayer.append(executeFunction)
                    tempLaveExeFunctionArr.remove(executeFunction)
                    tempCullExeFunctionArr.append(executeFunction)
            orderLayerArr.append(orderLayer)
        return orderLayerArr

    def makeRunAndRepFunctionArr(self, opsOrderDict):
        def getParentExeFunction(exeFunction, ordFunctionArr, repFunctionArr):
            for ordFunction in ordFunctionArr:
                if ordFunction['Child'] == exeFunction:
                    repFunctionArr.append(ordFunction['Parent'])
                    getParentExeFunction(ordFunction['Parent'], ordFunctionArr, repFunctionArr)
        exeFunctionArr = opsOrderDict['ExeFunctionArr']
        runFunctionArr = opsOrderDict['RunFunctionArr'] if 'RunFunctionArr' in opsOrderDict.keys() else exeFunctionArr
        repFunctionArr = opsOrderDict['RepFunctionArr'] if 'RepFunctionArr' in opsOrderDict.keys() else []
        noSLFunctionArr = opsOrderDict['NoSLFunctionArr'] if 'NoSLFunctionArr' in opsOrderDict.keys() else []
        for exeFunction in repFunctionArr:
            runFunctionArr.remove(exeFunction) if exeFunction in runFunctionArr else None

        if repFunctionArr == []:
            for exeFunction in runFunctionArr:
                getParentExeFunction(exeFunction, opsOrderDict['OrdFunctionArr'], repFunctionArr)
            repFunctionArr = list(set(repFunctionArr))
            tempRepFunctionArr = copy.deepcopy(repFunctionArr)
            for exeFunction in tempRepFunctionArr:
                if exeFunction in runFunctionArr:
                    repFunctionArr.remove(exeFunction)

        return runFunctionArr, repFunctionArr , noSLFunctionArr

    # ================================================== LoadRead ==================================================

    def saveRestltObject(self, opsInfo, executeFunction, functionRestlt, globalObjectDict , isSaveRestltObject = True, isSaveGlobalObject = True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]
        functionRestlt["ExeFunctionLDir"] = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion,str(opsRecordId), executeFunction)
        os.makedirs(functionRestlt["ExeFunctionLDir"]) if not os.path.isdir(functionRestlt["ExeFunctionLDir"]) else None
        if isSaveRestltObject == True :
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "FunctionRestlt.pickle"), 'wb') as f:
                pickle.dump(functionRestlt, f)
        if isSaveGlobalObject == True :
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "GlobalObjectDict.pickle"), 'wb') as f:
                pickle.dump(globalObjectDict, f)

        return functionRestlt , globalObjectDict

    def uploadRestltObject(self,opsInfo, executeFunction, functionRestlt, globalObjectDict , isUploadRestltObject = True, isUploadGlobalObject = True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl_Storage = SSHCtrl(host=os.getenv("SSH_IP"), port=int(os.getenv("SSH_PORT")), user=os.getenv("SSH_USER"),passwd=os.getenv("SSH_PASSWD")
                                  , timeout=60, printLog=False, isConnectSSH=True, isConnectSFTP=True)
        functionRestlt["ExeFunctionRDir"] = "{}/{}/{}/{}/{}".format(product, project, opsVersion, str(opsRecordId),executeFunction)
        sshCtrl_Storage.execCommand("mkdir -p /{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"), functionRestlt["ExeFunctionRDir"]))
        if isUploadRestltObject == True:
            sshCtrl_Storage.uploadFile("{}/{}".format(functionRestlt['ExeFunctionLDir'], "FunctionRestlt.pickle"),"/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),functionRestlt['ExeFunctionRDir'], "FunctionRestlt.pickle"))
        if isUploadGlobalObject == True:
            sshCtrl_Storage.uploadFile("{}/{}".format(functionRestlt['ExeFunctionLDir'], "GlobalObjectDict.pickle"),"/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),functionRestlt['ExeFunctionRDir'], "GlobalObjectDict.pickle"))
        del sshCtrl_Storage
        return functionRestlt , globalObjectDict

    def loadRestltObject(self, opsInfo, executeFunction, repOPSRecordId , functionRestlt, globalObjectDict , isLoadRestltObject = True, isLoadGlobalObject = True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]

        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId), executeFunction)
        if isLoadRestltObject == True:
            with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
                functionRestlt = pickle.load(fr)
        if isLoadGlobalObject == True:
            with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
                globalObjectDict = pickle.load(god)
        return functionRestlt, globalObjectDict

    def downloadRestltObject(self, opsInfo, executeFunction, repOPSRecordId , functionRestlt, globalObjectDict , isDownloadRestltObject = True, isDownloadGlobalObject = True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl_Storage = SSHCtrl(host=os.getenv("SSH_IP"), port=int(os.getenv("SSH_PORT")), user=os.getenv("SSH_USER"),passwd=os.getenv("SSH_PASSWD")
                                  , timeout=60, printLog=False, isConnectSSH=True, isConnectSFTP=True)
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId), executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId), executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        if isDownloadRestltObject== True:
            sshCtrl_Storage.downloadFile("/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"), exeFunctionRDir, "FunctionRestlt.pickle"),"{}/{}".format(exeFunctionLDir, "FunctionRestlt.pickle"))
        if isDownloadGlobalObject == True:
            sshCtrl_Storage.downloadFile("/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"), exeFunctionRDir, "GlobalObjectDict.pickle"),"{}/{}".format(exeFunctionLDir, "GlobalObjectDict.pickle"))
        del sshCtrl_Storage
        return functionRestlt, globalObjectDict