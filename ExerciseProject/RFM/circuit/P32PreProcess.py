import copy
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl


class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["R0_0_1"]
        mainDF = globalObjectFDict["ResultArr"][0]
        mainDF.columns = ["CustomerID","InvoiceNo","InvoiceDate","TotalPrice"]
        return {},{"ResultDF":mainDF}

