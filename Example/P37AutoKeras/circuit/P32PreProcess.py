
class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

    @classmethod
    def P0_0_2(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_2"])
        functionVersionInfo["Version"] = "P0_0_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict