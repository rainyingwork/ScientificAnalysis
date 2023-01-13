
class UseProduct() :

    @classmethod
    def UP0_0_4(self, functionInfo):
        import copy
        import torch
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_4"])
        functionVersionInfo["Version"] = "M0_0_4"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        model = globalObject['M0_0_4']["model"]
        xTestTensor = globalObject['P0_0_4']["xTestTensor"]
        yTestTensor = globalObject['P0_0_4']["yTestTensor"]

        model.eval()
        lossfunc = torch.nn.MSELoss()
        yTestPred = model(xTestTensor)  # 使用為訓練的資料進行損失函數驗證
        validLoss = lossfunc(yTestPred, yTestTensor)

        print(validLoss)

        return {}, {}

    @classmethod
    def UP0_0_5(self, functionInfo):
        import copy
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from sklearn.metrics import accuracy_score
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_5"])
        functionVersionInfo["Version"] = "UP0_0_5"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        modelFilePath = globalObject['M0_0_5']["ModelFilePath"]
        xTestTensor = globalObject['P0_0_5']["xTestTensor"]
        yTestTensor = globalObject['P0_0_5']["yTestTensor"]

        class NeuralNetwork(nn.Module):
            def __init__(self, inputSize):
                super(NeuralNetwork, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(inputSize, 100)
                    , nn.ReLU()
                    , nn.Linear(100, 100)
                    , nn.ReLU()
                    , nn.Linear(100, 2)
                )

            def forward(self, x):
                m = self.model(x)
                o = functional.log_softmax(m, dim=1)
                return o

        model = NeuralNetwork(xTestTensor.shape[1])
        model.load_state_dict(torch.load(modelFilePath))
        model.eval()
        testPred = model(xTestTensor)
        realTestPred = torch.exp(testPred)                              # 將log_softmax輸出轉為softmax輸出
        topPred, topClassTest = realTestPred.topk(1, dim=1)             # 取出最大值及其索引
        accTest = accuracy_score(yTestTensor, topClassTest) * 100       # 計算準確值
        print(f"accTest:{accTest:.2f}%")                                # acc_test:81.63%

        return {}, {}

    @classmethod
    def UP0_0_6(self, functionInfo):
        import copy
        import torch
        import torch.nn as nn
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_6"])
        functionVersionInfo["Version"] = "UP0_0_6"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        class NeuralNetwork(nn.Module):
            def __init__(self, inputSize):
                super(NeuralNetwork, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(inputSize, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 3)
                )

            def forward(self, x):
                return self.net(x)

        modelFilePath = globalObject['M0_0_6']["ModelFilePath"]
        xTestTensor = globalObject['P0_0_6']["xTestTensor"]
        yTestTensor = globalObject['P0_0_6']["yTestTensor"]

        model = NeuralNetwork(xTestTensor.shape[1])
        model.load_state_dict(torch.load(modelFilePath))

        testPred = model(xTestTensor)
        _, topClassTest = torch.max(testPred, dim=1)
        nCorrect = (yTestTensor.view(-1) == topClassTest).sum()        # 比對 yTestTensor.view(-1) 與 topClassTest 是否一致再加總
        print(f"Valid Acc:{nCorrect / len(xTestTensor):.4f}")          # 計算準確率

        return {}, {}

    @classmethod
    def UP0_0_7(self, functionInfo):
        import copy
        import torch
        import torch.nn as nn
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_7"])
        functionVersionInfo["Version"] = "UP0_0_7"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        modelFilePath = globalObject['M0_0_7']["ModelFilePath"]
        testDataLoader = globalObject['P0_0_7']["TestDataLoader"]

        class NeuralNetwork(nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                self.model = nn.Sequential(
                    # 圖片大小 28x28
                    nn.Conv2d(1, 16, 3, 1),         # 28x28x1 -> 26x26x16       16個3x3的卷積核
                    nn.ReLU(),                      # 26x26x16                  激活函數
                    nn.Conv2d(16, 32, 3, 1),        # 26x26x16 -> 24x24x32      32個3x3的卷積核
                    nn.ReLU(),                      # 24x24x32                  激活函數
                    nn.MaxPool2d(2),                # 24x24x32 -> 12x12x32      池化層
                    nn.Flatten(1),                  # 12x12x32 -> 4608          展平
                    nn.Linear(4608, 64),            # 12x12x32 -> 64            全連接層
                    nn.Dropout(0.10),               # 64 -> 64                  Dropout
                    nn.ReLU(),                      # 64 -> 64                  激活函數
                    nn.Linear(64, 10),              # 64 -> 10                  全連接層
                    nn.Dropout(0.25),               # 10 -> 10                  Dropout
                    nn.LogSoftmax(dim=1)            # 10 -> 10                  Softmax
                )
                # Conv2d：卷積層
                # Conv2d(in_channels,out_channels,kernel_size,stride,padding )
                # Conv2d(輸入通道數,輸出通道數,卷積核大小,步長,填充)
                # MaxPool2d：池化層 也可以使用 AvgPool2d
                # MaxPool2d(kernel_size,stride,padding)
                # MaxPool2d(池化核大小,步長,填充)
                # Flatten：展平層
                # Flatten(start_dim,end_dim)
                # Flatten(起始維度,結束維度)
                # Linear：全連接層
                # Linear(in_features,out_features)
                # Linear(輸入特徵數,輸出特徵數)
                # Dropout：Dropout層 也可以使用 BatchNorm2d
                # Dropout(p) ; BatchNorm2d(num_features)
                # Dropout(丟棄率) ; BatchNorm2d(特徵數)

            def forward(self, x):
                op = self.model(x)
                return op

        def test(model, device, testDataLoader, lossfunc):
            model.eval()
            loss, success = 0, 0
            with torch.no_grad():
                for x, y in testDataLoader:
                    x, y = x.to(device), y.to(device)
                    predProb = model(x)
                    loss += lossfunc(predProb, y).item()
                    pred = predProb.argmax(dim=1, keepdim=True)
                    success += pred.eq(y.view_as(pred)).sum().item()
                    datasize = len(testDataLoader.dataset)
                    overallLoss = loss / len(testDataLoader)
                    overallAcc = 100 * success / len(testDataLoader.dataset)
                    print('Loss: {:.4f}, Acc: {}/{} ({:.2f}%)'.format(overallLoss, success, datasize, overallAcc))

        lossfunc = nn.NLLLoss()

        device = torch.device('cpu')
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(modelFilePath))
        test(model, device, testDataLoader, lossfunc)
        sampleData, sampleTargets = next(iter(testDataLoader))
        predLabel = model(sampleData).max(dim=1)[1][10]
        print(f"Model prediction is : {predLabel}")
        print(f"Ground truth is : {sampleTargets[10]}")

        return {}, {}

    @classmethod
    def UP0_0_8(self, functionInfo):
        from PIL import Image
        import torch
        from torchvision import transforms
        from torchvision import models
        import numpy as np
        import pandas as pd

        preprocess = transforms.Compose([
            transforms.Resize(256),                                         # 縮放圖片長邊變成256
            transforms.CenterCrop(244),                                     # 從中心裁切出244x244的圖片
            transforms.ToTensor()                                           # 將圖片轉成Tensor，並把數值normalize到[0,1]
        ])

        resnetImageClassesDF = pd.read_csv("common/common/file/data/csv/ResnetImageClasses.csv", header=None)  # 對應表
        oriImg = Image.open("common/common/file/data/imgs/dog/dog.jpg")
        preImg = preprocess(oriImg)
        preImg = torch.unsqueeze(preImg, 0)

        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')         # 載入torchvision的預訓練模型
        model.eval()                                                        # 將resnetModel設為驗證
        predictList = model(preImg)                                         # 將圖片輸入 輸出成各個結果的機率

        predictNumpys = predictList.detach().numpy()                        # 轉為NumPy
        predictClass = np.argmax(predictNumpys, axis=1)                     # 找出最大值的索引
        predictLabel = resnetImageClassesDF.iloc[predictClass].values       # 找出對應的類別
        print(predictLabel)
        score = torch.nn.functional.softmax(predictList, dim=1)[0] * 100    # 列出所有對應標籤的百分比
        print(f"Score:{score[predictClass].item():.2f}")                    # 找出對應標籤的百分比

        return {}, {}

    @classmethod
    def UP0_0_9(self, functionInfo):
        import torch
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        import torchvision
        import numpy

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_9"])
        functionVersionInfo["Version"] = "UP0_0_9"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        trainDataSet = globalObject['P0_0_9']["TrainDataSet"]
        verifyDataSet = globalObject['P0_0_9']["VerifyDataSet"]
        trainDataLoader = globalObject['P0_0_9']["TrainDataLoader"]
        verifyDataLoader = globalObject['P0_0_9']["VerifyDataLoader"]

        model = torch.load("Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_9/model.pt")

        inputs, classes = next(iter(verifyDataLoader))  # 取得一批圖像
        className = verifyDataSet.classes  # 建立類別名稱列表

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        inputs = inputs.to(device)
        outputs = model(inputs)  # 預測輸出
        _, preds = torch.max(outputs, 1)
        title = [className[x] for x in preds]

        out = torchvision.utils.make_grid(inputs)           # 顯示圖像
        out = out.numpy().transpose((1, 2, 0))              # 轉換成numpy
        mean = numpy.array([0.485, 0.456, 0.406])           # 轉換成標準圖像
        std = numpy.array([0.229, 0.224, 0.225])            # 轉換成標準圖像
        out = std * out + mean                              # 取消在transforms.Normalize()中的標準化
        out = numpy.clip(out, 0, 1)                         # 將圖像限制在0~1之間

        print(title)
        print(out)

        return {}, {}

    @classmethod
    def UP0_0_10(self, functionInfo):
        import torch
        from torch import nn
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        # 轉為張量與作正規化
        transform = transforms.Compose([  # 轉為張量與作正規化
            transforms.ToTensor(),  # 轉為張量
            transforms.Normalize(  # 正規化
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
        ])

        testData = datasets.CIFAR10('common/common/file/data/imgs/cifar10/test', train=False, download=True,transform=transform)

        batchSize = 100
        testDataLoader = DataLoader(testData, batch_size=batchSize)
        print(len(testDataLoader))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Deivce:{device}")

        import Example.P34PyTorch.package.cifar10_model as cifar10Model
        modelFile = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_10/cifar10_model.pt"

        model = cifar10Model.CNN()
        model.load_state_dict(torch.load(modelFile))

        model = model.to(device)

        lossfunc = nn.NLLLoss()

        numCorrect = 0.0  # 正確數量
        testLoss = 0
        model.eval()  # 模型驗證
        for x, y in testDataLoader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = lossfunc(pred, y)  # 計算損失函數
            testLoss += loss.item()
            _, predicted = torch.max(pred, 1)
            numCorrect += (predicted == y).float().sum()

        testLoss = testLoss / len(testDataLoader)
        numCorrect = numCorrect / (len(testDataLoader) * batchSize)
        print(f"TestLoss: {testLoss:.3f}, Correct: {numCorrect:.3f}")

        return {}, {}

    @classmethod
    def UP1_0_1(self, functionInfo):
        import copy
        import torch
        from torch import nn
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP1_0_1"])
        functionVersionInfo["Version"] = "UP1_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        testData = globalObject["P1_0_1"]["TestData"]
        pytorchModelFilePath = globalObject["M1_0_1"]["PyTorchModelFilePath"]
        itemNo = functionVersionInfo["ItemNo"]

        class NeuralNetwork(nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                # 建立類神經網路各層
                self.flatten = nn.Flatten()  # 轉為一維向量
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(28 * 28, 512),  # 線性轉換
                    nn.ReLU(),  # ReLU 轉換
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                )

            def forward(self, x):
                # 定義資料如何通過類神經網路各層
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        model = NeuralNetwork()
        model.load_state_dict(torch.load(pytorchModelFilePath))
        classes = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]

        # 將模型設定為驗證模式
        model.eval()

        # 模型設定驗證模式
        x, y = testData[itemNo][0], testData[itemNo][1]
        with torch.no_grad():  # 不要計算參數梯度
            # 以模型進行預測
            pred = model(x)
            # 整理測試結果
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Real："{actual}" / Pred："{predicted}"')

        return {}, {}
