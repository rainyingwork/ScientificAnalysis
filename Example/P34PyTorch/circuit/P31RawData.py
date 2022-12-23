
class RawData() :

    @classmethod
    def R0_0_4(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('common/common/file/data/csv/YearPredictionMSD.csv', nrows=50000, header=None)

        return {}, {"ResultDF":mainDF}

    @classmethod
    def R0_0_5(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('common/common/file/data/csv/UCICreditCard.csv')
        print(mainDF.shape)  # (30000, 25)

        return {}, {"ResultDF":mainDF}

    @classmethod
    def R0_0_6(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('Example/P34PyTorch/file/data/iris.csv')
        print(mainDF.shape)  # (30000, 25)

        return {}, {"ResultDF": mainDF}

    @classmethod
    def R1_0_1(self, functionInfo):
        import copy
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R1_0_1"])
        functionVersionInfo["Version"] = "R1_0_1"
        # 訓練資料撈取
        trainData = datasets.FashionMNIST(
            root="UnitTest/PyTorch/file/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        # 測試資料撈取
        testData = datasets.FashionMNIST(
            root="UnitTest/PyTorch/file/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        return {} , {"TrainData":trainData,"TestData":testData}

