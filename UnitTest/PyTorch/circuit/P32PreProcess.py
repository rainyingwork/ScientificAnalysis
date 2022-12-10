
class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        # 為資料處理保留位置
        trainData = globalObject[functionVersionInfo["DataVersion"]]["TrainData"]
        testData = globalObject[functionVersionInfo["DataVersion"]]["TestData"]
        return {}, {"TrainData":trainData,"TestData":testData}

