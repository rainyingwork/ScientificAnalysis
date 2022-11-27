import os , copy , pprint
import threading
import time
from queue import Queue
import pickle

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
        orderFunctionLayerArr = self.makeOrderLayerArr(opsInfo)
        replyOPSRecordId = opsInfo['OPSOrderJson']["ReplyOPSRecordId"] if "ReplyOPSRecordId" in opsInfo['OPSOrderJson'].keys() else 0
        replyExecuteArr =  opsInfo['OPSOrderJson']["ReplyExecuteArr"] if "ReplyExecuteArr" in opsInfo['OPSOrderJson'].keys() else []
        for orderFunctionLayer in orderFunctionLayerArr :
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer :
                if executeFunction in replyExecuteArr :
                    thread = threading.Thread(target=self.replyExecuteFunction, args=(executeFunction,opsInfo,replyOPSRecordId,threadQueue))
                else :
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
            functionRestlt["ExeFunctionLDir"] = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
            functionRestlt["ExeFunctionRDir"] = "Product={}/Project={}/OPSVersion={}/OPSRecordId={}/EXEFunction={}"\
                                                .format(product,project,opsVersion,opsRecordId,executeFunction)
            os.makedirs(functionRestlt["ExeFunctionLDir"]) if not os.path.isdir(functionRestlt["ExeFunctionLDir"]) else None
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "FunctionRestlt.pickle"), 'wb') as f:
                pickle.dump(functionRestlt, f)
            with open("{}/{}".format(functionRestlt["ExeFunctionLDir"], "GlobalObjectDict.pickle"), 'wb') as f:
                pickle.dump(globalObjectDict, f)
            opsDetailEntityCtrl = OPSDetailEntity()
            functionInfo = {}
            functionInfo["OPSRecordId"] = opsInfo["OPSRecordId"]
            functionInfo["Exefunction"] = executeFunction
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

    def replyExecuteFunction(self,executeFunction, opsInfo , replyOPSRecordId , threadQueue):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(replyOPSRecordId),executeFunction)
        with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
            functionRestlt = pickle.load(fr)
        with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
            globalObjectDict = pickle.load(god)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  Reply Function , Version is {} , ReplyOPSDetailID is {} ".format(executeFunction,str(replyOPSRecordId)))

    def makeOrderLayerArr(self, opsInfo):
        opsOrderDict = opsInfo['OPSOrderJson']
        executeFunctionArr = opsOrderDict['ExecuteArr']
        orderFunctionArr = opsOrderDict['OrderArr']
        orderLayerArr = []
        tempLaveExecuteFunctionArr = copy.deepcopy(executeFunctionArr)
        tempCullExecuteFunctionArr = []
        while len(tempLaveExecuteFunctionArr) != 0:
            orderLayer = []
            tempInitLaveExecuteFunctionArr = copy.deepcopy(tempLaveExecuteFunctionArr)
            tempInitCullExecuteFunctionArr = copy.deepcopy(tempCullExecuteFunctionArr)
            for executeFunction in tempInitLaveExecuteFunctionArr:
                isNotCullParent = False
                for orderFunction in orderFunctionArr:
                    if orderFunction['Child'] == executeFunction and orderFunction['Parent'] not in tempInitCullExecuteFunctionArr:
                        isNotCullParent = True
                if isNotCullParent == False:
                    orderLayer.append(executeFunction)
                    tempLaveExecuteFunctionArr.remove(executeFunction)
                    tempCullExecuteFunctionArr.append(executeFunction)
            orderLayerArr.append(orderLayer)
        return orderLayerArr


