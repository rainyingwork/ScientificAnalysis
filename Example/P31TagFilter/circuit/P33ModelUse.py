import copy
import pprint

class ModelUse() :

    @classmethod
    def M0_0_1(self, functionInfo):
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        functionVersionInfo["ResultIDArr"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["ResultIDArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["P0_0_1"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

