
class PreProcess() :

    @classmethod
    def P0_2_1(self, functionInfo):
        import copy , numpy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_2_1"])
        functionVersionInfo["Version"] = "P0_2_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        movieDF = globalObject["R0_2_1"]["ResultArr"][0]
        ratingDF = globalObject["R0_2_2"]["ResultArr"][0]
        movieDF.columns = ['MovieID','MovieTitle']
        ratingDF.columns = ['MovieID','UserID','Rating']

        mainDF = movieDF.merge(ratingDF, on='MovieID')

        return {}, {"ResultDF": mainDF}

