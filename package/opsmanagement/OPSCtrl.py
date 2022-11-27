import os , copy , pprint
import threading
import time
from queue import Queue
import pickle
from dotenv import load_dotenv
from package.common.osbasic.SSHCtrl import SSHCtrl


class OPSCtrl:

    def __init__(self):
        load_dotenv(dotenv_path="env/ssh.env")
        self.sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

    def executeOPS(self, opsInfo):
        allGlobalObjectDict = {}
        opsInfo["GlobalObject"] = id(allGlobalObjectDict)
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]
        print("Start OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product , project, opsVersion,opsRecordId))
        opsOrderDict = self.makeCompleteOPSOrderDict(opsInfo['OPSOrderJson'])
        repOPSRecordId = opsOrderDict["RepOPSRecordId"] if "RepOPSRecordId" in opsOrderDict.keys() else 0
        repFunctionArr = opsOrderDict["RepFunctionArr"] if "RepFunctionArr" in opsOrderDict.keys() else []
        runFunctionArr = opsOrderDict["RunFunctionArr"] if "RunFunctionArr" in opsOrderDict.keys() else []
        for orderFunctionLayer in opsOrderDict["OrderLayerArr"] :
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer :
                if executeFunction in repFunctionArr :
                    thread = threading.Thread(target=self.replyExecuteFunction, args=(executeFunction,opsInfo,repOPSRecordId,threadQueue))
                    thread.daemon = True
                    thread.start(), time.sleep(0.5)
                    threadList.append(thread)
                if executeFunction in runFunctionArr :
                    thread = threading.Thread(target=self.runExecuteFunction, args=(executeFunction,opsInfo,threadQueue))
                    thread.daemon = True
                    thread.start() , time.sleep(0.5)
                    threadList.append(thread)
            for thread in threadList:
                thread.join()
            for _ in threadList:
                functionDict = threadQueue.get()
                executeFunction = functionDict["ExecuteFunction"]
                opsInfo["ResultJson"][executeFunction] = functionDict["FunctionRestlt"]
                allGlobalObjectDict[executeFunction] = functionDict["GlobalObjectDict"]
        print("End OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(product , project, opsVersion,opsRecordId))

    def runExecuteFunction(self,executeFunction, opsInfo, threadQueue):

        def makeExecuteFunctionInfo(opsInfo, executeFunction, functionRestlt, globalObjectDict):
            product = opsInfo["Product"]
            project = opsInfo["Project"]
            opsVersion = opsInfo["OPSVersion"]
            opsRecordId = opsInfo["OPSRecordId"]
            from package.opsmanagement.entity.OPSDetailEntity import OPSDetailEntity
            functionRestlt["ExeFunctionLDir"] = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(opsRecordId),executeFunction)
            functionRestlt["ExeFunctionRDir"] = "{}/{}/{}/{}/{}".format(product,project,opsVersion,str(opsRecordId),executeFunction)
            os.makedirs(functionRestlt["ExeFunctionLDir"]) if not os.path.isdir(functionRestlt["ExeFunctionLDir"]) else None
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "FunctionRestlt.pickle"), 'wb') as f:
                pickle.dump(functionRestlt, f)
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "GlobalObjectDict.pickle"), 'wb') as f:
                pickle.dump(globalObjectDict, f)
            self.sshCtrl.execCommand("mkdir -p /Storage/OPSData/{}".format(functionRestlt["ExeFunctionRDir"]))
            self.sshCtrl.uploadFile("{}/{}".format(functionRestlt['ExeFunctionLDir'],"FunctionRestlt.pickle")
                                    , "/Storage/OPSData/{}/{}".format(functionRestlt['ExeFunctionRDir'],"FunctionRestlt.pickle") )
            self.sshCtrl.uploadFile("{}/{}".format(functionRestlt['ExeFunctionLDir'],"GlobalObjectDict.pickle")
                                    , "/Storage/OPSData/{}/{}".format(functionRestlt['ExeFunctionRDir'],"GlobalObjectDict.pickle") )
            opsDetailEntityCtrl = OPSDetailEntity()
            functionInfo = {}
            functionInfo["OPSRecordId"] = opsInfo["OPSRecordId"]
            functionInfo["ExeFunction"] = executeFunction
            functionInfo["ParameterJson"] = opsInfo["ParameterJson"][executeFunction] if executeFunction in opsInfo["ParameterJson"].keys() else {}
            functionInfo["ResultJson"] = functionRestlt
            opsDetailEntityCtrl.setEntity(opsDetailEntityCtrl.makeOPSRecordEntityByFunctionInfo(functionInfo))
            opsDetailEntityCtrl.insertEntity()
            return opsDetailEntityCtrl.getEntityId()

        product = opsInfo["Product"]
        project = opsInfo["Project"]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        print("  Start Function , Version is {}  ".format(executeFunction))
        functionRestlt, globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
        opsDetailId = makeExecuteFunctionInfo(opsInfo, executeFunction, functionRestlt,globalObjectDict)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  End Function , Version is {} , OPSDetailID is {} ".format(executeFunction,opsDetailId))

    def replyExecuteFunction(self,executeFunction, opsInfo , repOPSRecordId , threadQueue):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId), executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        self.sshCtrl.downloadFile("/Storage/OPSData/{}/{}".format(exeFunctionRDir,"FunctionRestlt.pickle"),"{}/{}".format(exeFunctionLDir, "FunctionRestlt.pickle"))
        self.sshCtrl.downloadFile("/Storage/OPSData/{}/{}".format(exeFunctionRDir,"GlobalObjectDict.pickle"),"{}/{}".format(exeFunctionLDir, "GlobalObjectDict.pickle"))
        with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
            functionRestlt = pickle.load(fr)
        with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
            globalObjectDict = pickle.load(god)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  Reply Function , Version is {} , ReplyOPSDetailID is {} ".format(executeFunction,str(repOPSRecordId)))

    def makeCompleteOPSOrderDict(self, opsOrderDict):
        opsOrderDict["OrderLayerArr"] = self.makeOrderLayerArr(opsOrderDict)
        opsOrderDict['RunFunctionArr'] , opsOrderDict['RepFunctionArr']  = self.makeRunAndRepFunctionArr(opsOrderDict)
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

        return runFunctionArr, repFunctionArr

