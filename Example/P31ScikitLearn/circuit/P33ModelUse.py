
class ModelUse() :

    @classmethod
    def M0_0_1(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

