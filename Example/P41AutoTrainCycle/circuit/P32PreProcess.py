
class PreProcess() :

    @classmethod
    def P0_1_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_1_X"])
        functionVersionInfo["Version"] = "P0_1_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        _ , globalObjectDict['ResultArr'][0] = preProcessFunction.filterXAllZeroColumn(functionVersionInfo, globalObjectDict['ResultArr'][0])
        return resultObject , globalObjectDict

    @classmethod
    def P0_11_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_11_X"])
        functionVersionInfo["Version"] = "P0_11_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        _ , globalObjectDict['ResultArr'][0] = preProcessFunction.filterXAllZeroColumn(functionVersionInfo, globalObjectDict['ResultArr'][0])
        return resultObject , globalObjectDict

    @classmethod
    def P0_12_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_12_X"])
        functionVersionInfo["Version"] = "P0_12_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        _ , globalObjectDict['ResultArr'][0] = preProcessFunction.filterXAllZeroColumn(functionVersionInfo, globalObjectDict['ResultArr'][0])
        return resultObject , globalObjectDict

    @classmethod
    def P0_20_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_20_X"])
        functionVersionInfo["Version"] = "P0_20_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        _ , globalObjectDict['ResultArr'][0] = preProcessFunction.filterXAllZeroColumn(functionVersionInfo, globalObjectDict['ResultArr'][0])
        return resultObject , globalObjectDict

    @classmethod
    def P0_29_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_29_X"])
        functionVersionInfo["Version"] = "P0_29_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        resultArr = []
        for dataVersionName in functionVersionInfo["DataVersion"] :
            oriResultArr = globalObject[dataVersionName]["ResultArr"]
            for oriResult in oriResultArr :
                resultArr.append(oriResult)
        functionVersionInfo["ResultArr"] = resultArr
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

