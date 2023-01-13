
class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = preProcessFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

