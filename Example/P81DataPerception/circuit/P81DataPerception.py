
class DataPerception():
    @classmethod
    def DP0_0_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.dataperception.DataPercoptionFunction import DataPercoptionFunction
        dataPercoptionFunction = DataPercoptionFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["DP0_0_1"])
        functionVersionInfo["Version"] = "DP0_0_1"
        resultObject, globalObjectDict = dataPercoptionFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def DP0_1_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.dataperception.DataPercoptionFunction import DataPercoptionFunction
        dataPercoptionFunction = DataPercoptionFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["DP0_1_1"])
        functionVersionInfo["Version"] = "DP0_1_1"
        resultObject, globalObjectDict = dataPercoptionFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def DP0_1_2(self, functionInfo):
        import copy
        from package.artificialintelligence.common.dataperception.DataPercoptionFunction import DataPercoptionFunction
        dataPercoptionFunction = DataPercoptionFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["DP0_1_2"])
        functionVersionInfo["Version"] = "DP0_1_2"
        resultObject, globalObjectDict = dataPercoptionFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict