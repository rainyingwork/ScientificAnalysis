import copy , pprint
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

class RawData() :

    @classmethod
    def R0_0_1(self, functionInfo):
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

