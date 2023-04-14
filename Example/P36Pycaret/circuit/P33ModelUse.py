
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
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_2(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_2"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_3(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_3"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_3"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        ModelUseFunction.insertOverwriteAnalysisData("Example","P36Pycaret","V0_0_3","20220101",globalObjectDict["ResultArr"][0], useType= 'IO')
        return resultObject, globalObjectDict