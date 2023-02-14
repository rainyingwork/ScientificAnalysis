
class Other() :

    @classmethod
    def OTHER0_0_1(self, functionInfo):
        import copy
        from package.opsmanagement.common.funcresult.FuncResultFunction import FuncResultFunction
        funcResultFunction = FuncResultFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["OTHER0_0_1"])
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Function"] = "OTHER0_0_1"
        resultObject, globalObjectDict = funcResultFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

