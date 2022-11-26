import os , copy
import threading
from queue import Queue

class OPSCtrl:

    def __init__(self):
        pass

    def executeOPS(self, opsInfo):
        allGlobalObjectDict = {}
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        print("start OPS , product is {} , project is {} , version is {}".format(product , project, opsVersion))
        orderFunctionLayerArr = self.makeOrderLayerArr(opsInfo)
        for orderFunctionLayer in orderFunctionLayerArr :
            threadList = []
            threadQueue = Queue()
            for executeFunction in orderFunctionLayer :
                thread = threading.Thread(target=self.runExecuteFunction, args=(executeFunction,opsInfo,threadQueue))
                thread.start()
                threadList.append(thread)
            for thread in threadList:
                thread.join()
            for _ in threadList:
                functionDict = threadQueue.get()
                executeFunction = functionDict["ExecuteFunction"]
                opsInfo["ResultJson"][executeFunction] = functionDict["FunctionRestlt"]
                allGlobalObjectDict[executeFunction] = functionDict["GlobalObjectDict"]
        print("start OPS , product is {} , project is {} , version is {}".format(product , project, opsVersion))

    def runExecuteFunction(self,executeFunction, opsInfo, threadQueue):
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        print("  start function , version is {} ".format(executeFunction))
        functionRestlt, globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
        threadQueue.put({
            "ExecuteFunction": executeFunction
            , "FunctionRestlt": functionRestlt
            , "GlobalObjectDict": globalObjectDict
        })
        print("  end function , version is {}".format(executeFunction))

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
