
class Docker():

    @classmethod
    def D1_1_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_1_0")

    @classmethod
    def D1_1_1(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_1_1")

    @classmethod
    def D1_1_2(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_1_2")

    @classmethod
    def D1_2_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_2_0")

    @classmethod
    def D1_2_1(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_2_1")

    @classmethod
    def D1_2_2(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_2_2")

    @classmethod
    def D1_6_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D1_6_0")

    @classmethod
    def D2_1_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_1_0")

    @classmethod
    def D2_1_1(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_1_1")

    @classmethod
    def D2_2_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_2_0")

    @classmethod
    def D2_3_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_3_0")

    @classmethod
    def D2_6_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_6_0")

    @classmethod
    def D2_8_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_8_0")

    @classmethod
    def D2_9_0(self, functionInfo):
        return self.__DX_X_X(functionInfo, "D2_9_0")

    @classmethod
    def __DX_X_X(self, functionInfo, functionVersion):
        import copy
        from package.systemengineer.common.docker.DockerFunction import DockerFunction
        dockerFunction = DockerFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"][functionVersion])
        functionVersionInfo["Version"] = functionVersion
        resultObject, globalObjectDict = dockerFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict