
class Docker():

    @classmethod
    def D1_1_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_1_0")

    @classmethod
    def D1_1_1(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_1_1")

    @classmethod
    def __DX_X_X(self, functionInfo, functionVersion):
        import copy
        from package.systemengineer.common.docker.DockerFunction import DockerFunction
        dockerFunction = DockerFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"][functionVersion])
        functionVersionInfo["Version"] = functionVersion
        resultObject, globalObjectDict = dockerFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

