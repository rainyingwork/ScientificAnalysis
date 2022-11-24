import os , copy , pprint

class OPSCtrl:

    def __init__(self):
        pass

    def executeOPS(self, opsInfo):
        allGlobalObjectDict = {}
        product = opsInfo["Product"]
        project = opsInfo["Project"]
        opsVersion = opsInfo["OPSVersion"]
        print("start OPS , product is {} , project is {} , version is {}".format(product , project, opsVersion))
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        orderFunctionLayerArr = self.makeOrderLayerArr(opsInfo)
        for orderFunctionLayer in orderFunctionLayerArr :
            for executeFunction in orderFunctionLayer :
                print("  start function , version is {} ".format(executeFunction))
                functionRestlt , globalObjectDict = eval(f"circuitMain.{executeFunction}({opsInfo})")
                opsInfo["ResultJson"][executeFunction] = functionRestlt
                allGlobalObjectDict[executeFunction] = globalObjectDict
                print("  end function , version is {}".format(executeFunction))
        print("start OPS , product is {} , project is {} , version is {}".format(product , project, opsVersion))

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
