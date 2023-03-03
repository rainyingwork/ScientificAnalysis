import os , copy , shutil
import threading
import time , pprint
from queue import Queue
import pickle
from dotenv import load_dotenv

class OPSCtrl:

    def __init__(self):
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
        pprint.pprint(opsInfo["ResultJson"])

    def runExecuteFunction(self,executeFunction, opsInfo, threadQueue,isSL=True):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        print("  Start Function , Version is {}  ".format(executeFunction))
        functionRestlt, globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
        if isSL == True:
            functionRestlt, globalObjectDict = self.saveRestltObject(opsInfo, executeFunction, functionRestlt, globalObjectDict)
        opsDetailId = None
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  End Function , Version is {} , OPSDetailID is {} ".format(executeFunction,opsDetailId))

    def replyExecuteFunction(self,executeFunction, opsInfo , repOPSRecordId , threadQueue):
        functionRestlt, globalObjectDict = self.loadRestltObject(opsInfo, executeFunction, repOPSRecordId, None, None)
        functionRestlt, globalObjectDict = self.saveRestltObject(opsInfo, executeFunction, functionRestlt, globalObjectDict)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  Reply Function , Version is {} , ReplyOPSDetailID is {} ".format(executeFunction,str(repOPSRecordId)))

    # ================================================== CompleteOPSOrderDict ==================================================
    def makeExecuteFunctionInfo(self, opsInfo, executeFunction, functionRestlt, globalObjectDict):
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
