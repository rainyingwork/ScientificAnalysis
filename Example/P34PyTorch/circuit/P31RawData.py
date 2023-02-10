
class RawData() :

    @classmethod
    def R0_0_4(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('common/common/file/data/csv/YearPredictionMSD.csv', nrows=50000, header=None)
        print(mainDF.shape)

        return {}, {"ResultDF":mainDF}

    @classmethod
    def R0_0_5(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('common/common/file/data/csv/UCICreditCard.csv')
        print(mainDF.shape)

        return {}, {"ResultDF":mainDF}

    @classmethod
    def R0_0_6(self, functionInfo):
        import pandas

        mainDF = pandas.read_csv('common/common/file/data/csv/Iris.csv')
        print(mainDF.shape)  # (30000, 25)

        return {}, {"ResultDF": mainDF}

    @classmethod
    def R0_0_7(self, functionInfo):
        import torch
        from torchvision import datasets, transforms

        torch.manual_seed(0)                                                                # 設定隨機種子

        transform = transforms.Compose([transforms.ToTensor(), ])
        trainDataSet = datasets.MNIST('common/common/file/data/imgs', train=True, download=True, transform=transform)
        testDataSet = datasets.MNIST('common/common/file/data/imgs', train=False, transform=transform)

        return {}, {"TrainDataSet": trainDataSet,"TestDataSet": testDataSet}

    @classmethod
    def R1_0_1(self, functionInfo):
        import copy
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R1_0_1"])
        functionVersionInfo["Version"] = "R1_0_1"
        # 訓練資料撈取
        trainData = datasets.FashionMNIST(
            root="Example/P34PyTorch/file/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        # 測試資料撈取
        testData = datasets.FashionMNIST(
            root="Example/P34PyTorch/file/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        return {} , {"TrainData":trainData,"TestData":testData}

