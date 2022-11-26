import copy
import pprint
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

class ModelUse() :

    @classmethod
    def M0_0_1(self, functionInfo):
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_2(self, functionInfo):
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_2"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["M0_0_2"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_2"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_2"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_3(self, functionInfo):
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_3"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_3"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["M0_0_3"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_3"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_3"]["DataVersion"]]["MakeDataInfo"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict