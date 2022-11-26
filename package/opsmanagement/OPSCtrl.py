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
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        opsRecordId = opsInfo["OPSRecordId"]
        print("start OPS , product is {} , project is {} , version is {} , opsrecordid is {}".format(product , project, opsVersion,opsRecordId))
        orderFunctionLayerArr = self.makeOrderLayerArr(opsInfo)
        for orderFunctionLayer in orderFunctionLayerArr :
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer :
                thread = threading.Thread(target=self.runExecuteFunction, args=(executeFunction,opsInfo,threadQueue))
                thread.start() , time.sleep(0.5)
                threadList.append(thread)
            for thread in threadList:
                thread.join()
            for _ in threadList:
                functionDict = threadQueue.get()
                executeFunction = functionDict["ExecuteFunction"]
                opsInfo["ResultJson"][executeFunction] = functionDict["FunctionRestlt"]
                allGlobalObjectDict[executeFunction] = functionDict["GlobalObjectDict"]
        print("start OPS , product is {} , project is {} , version is {} , opsrecordid is {}".format(product , project, opsVersion,opsRecordId))

    def runExecuteFunction(self,executeFunction, opsInfo, threadQueue):

        def makeExecuteFunctionInfo(opsInfo, executeFunction, functionRestlt, globalObjectDict):
            product = opsInfo["Product"]
            project = opsInfo["Project"]
            opsVersion = opsInfo["OPSVersion"]
            opsRecordId = opsInfo["OPSRecordId"]
            from package.opsmanagement.entity.OPSDetailEntity import OPSDetailEntity
            functionRestlt["ExeFunctionLDir"] = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
            functionRestlt["exeFunctionRDir"] = "Product={}/Project={}/OPSVersion={}/OPSRecordId={}/EXEFunction={}"\
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
        print("  start function , version is {}  ".format(executeFunction))
        functionRestlt, globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
        opsDetailId = makeExecuteFunctionInfo(opsInfo, executeFunction, functionRestlt,globalObjectDict)
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  end function , version is {} , opsdetailid is {} ".format(executeFunction,opsDetailId))

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


