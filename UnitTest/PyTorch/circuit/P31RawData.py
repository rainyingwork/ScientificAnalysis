
class RawData() :

    @classmethod
    def R0_0_1(self, functionInfo):
        import copy
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
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

