
class ModelUse() :

    @classmethod
    def M0_2_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_2_1"])
        functionVersionInfo["Version"] = "M0_2_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_2_1"]
        mainDF = globalObjectFDict["ResultDF"]

        mainPivotDF = mainDF.pivot_table(index = ["UserID"],columns = ["MovieTitle"],values = "Rating")

        return {}, {"ResultDF": mainPivotDF}

