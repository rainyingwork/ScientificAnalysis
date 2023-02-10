
class InputCtrl:
    def makeParametersData(self,argv):
        parameterArgv = argv
        parametersData = {}
        if len(parameterArgv) >= 1:
            parameterName = None
            parameterValues = []
            for parameter in parameterArgv[1:]:
                if parameter.find("--") == 0:
                    if parameterName != None:
                        parametersData[parameterName] = parameterValues
                    parameterValues = []
                    parameterName = parameter.replace("--", "")
                else:
                    parameterValues.append(parameter)
            parametersData[parameterName] = parameterValues
        return parametersData
