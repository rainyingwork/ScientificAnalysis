
class Docker():

    @classmethod
    def D0_0_1(self, functionInfo):
        import copy
        from package.systemengineer.common.docker.DockerFunction import DockerFunction
        dockerFunction = DockerFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["D0_0_1"])
        functionVersionInfo["Version"] = "D0_0_1"
        resultObject, globalObjectDict = dockerFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict


