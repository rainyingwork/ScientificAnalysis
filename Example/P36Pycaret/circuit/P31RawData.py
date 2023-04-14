
class RawData() :

    @classmethod
    def R0_0_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

    @classmethod
    def R0_0_2(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_2"])
        functionVersionInfo["Version"] = "R0_0_2"
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_0_3(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_3"])
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict