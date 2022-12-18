
class SYS ():

    @classmethod
    def SYS1_0_1(self, functionInfo):
        import copy
        from package.systemengineer.common.sys.SYSFunction import SYSFunction
        sysFunction = SYSFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["SYS1_0_1"])
        functionVersionInfo["Version"] = "SYS1_0_1"
        return sysFunction.executionFunctionByFunctionType(functionVersionInfo)

    @classmethod
    def SYS1_0_2(self, functionInfo):
        import copy
        from package.systemengineer.common.sys.SYSFunction import SYSFunction
        sysFunction = SYSFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["SYS1_0_2"])
        functionVersionInfo["Version"] = "SYS1_0_2"
        return sysFunction.executionFunctionByFunctionType(functionVersionInfo)