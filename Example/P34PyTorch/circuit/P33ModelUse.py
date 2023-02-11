class ModelUse():

    @classmethod
    def M0_0_2(self, functionInfo):
        import copy
        import torch
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_2"])
        functionVersionInfo["Version"] = "M0_0_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        x ,y = globalObject['P0_0_2']["x"] , globalObject['P0_0_2']["y"]

        torch.manual_seed(0) # 設定隨機種子

        # 先隨機給三個數字，經由梯度下降算出正確的數字
        # 為了能夠進行反向傳播計算，必須將該物件的 requires_grad 方法設置為 True
        wPred = torch.randn(3, requires_grad=True)
        # 損失函數值，該值越高學習越快，該值越低學習越慢，但該值越高會越容易跳過局部最優
        lr = 0.01

        lossValueList = []                                                      # 損失值紀錄
        epochs = 200                                                            # 重複計算次數
        for epoch in range(epochs + 1):
            yPred = torch.mv(x, wPred)
            loss = torch.mean((y - yPred) ** 2)                                 # 計算MSE
            wPred.grad = None                                                   # 清除上一次計算的梯度值
            loss.backward()                                                     # loss 向輸入側進行反向傳播，這時候 w_pred.grad 就會有值
            wPred.data = wPred.data - lr * wPred.grad.data                      # 梯度下降更新 原本資料 - 損失函數 * 差異梯度
            lossValueList.append(loss.item())                                   # 記錄該次的loss值
            if (epoch) % 50 == 0 :                                              # 印出50倍數代的相關數據
                print(f"Epoch:{epoch}, Loss: {loss.item():.3f}")

        # 印出損失值過程
        print(lossValueList)
        # 印出最終預設的值
        print(wPred.data)
        return {}, {}

    @classmethod
    def M0_0_3(self, functionInfo):
        import copy
        import torch.nn as nn
        import torch.optim as optim
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_3"])
        functionVersionInfo["Version"] = "M0_0_3"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        x, y = globalObject['P0_0_3']["x"], globalObject['P0_0_3']["y"]

        # 使用線性神經網路 輸入3 輸出1 偏置設置為False
        # 何謂偏置 -> 基本來說設定 False，至於怎麼用可能要問一下數學家..XD
        model = nn.Sequential(
            nn.Linear(3, 1, bias=False)
        )

        # 損失函數：使用MSELoss 與 學習函數：使用Adam
        lossfunc = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        lossList = []
        epochs = 101
        for epoch in range(epochs):
            y_pred = model(x)
            loss = lossfunc(y_pred, y)
            lossList.append(loss.item())
            optimizer.zero_grad()                                               # 清除上一次計算的梯度值
            loss.backward()                                                     # loss 向輸入側進行反向傳播
            optimizer.step()                                                    # 做梯度下降更新
            if epoch % 20 == 0:                                                 # 印出20倍數代的相關數據
                print(f"Epoch:{epoch}, Loss:{loss.item():.3f}")

        # 印出損失值過程
        print(lossList)
        # 印出最終預設的值
        print(list(model.parameters()))

        return {}, {}

    @classmethod
    def M0_0_4(self, functionInfo):
        import copy
        import torch
        import torch.nn as nn
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_4"])
        functionVersionInfo["Version"] = "M0_0_4"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        # 設定隨機種子
        torch.manual_seed(0)

        xTrainTensor = globalObject['P0_0_4']["xTrainTensor"]
        yTrainTensor = globalObject['P0_0_4']["yTrainTensor"]
        xDevTensor = globalObject['P0_0_4']["xDevTensor"]
        yDevTensor = globalObject['P0_0_4']["yDevTensor"]

        # 建立神經網路
        model = nn.Sequential(
            nn.Linear(xTrainTensor.shape[1], 200),  # 輸入層 根據xTrainTensor的shape[1]來建立輸入
            nn.ReLU(),  # 激勵函數
            nn.Linear(200, 50),  # 中間層
            nn.ReLU(),  # 激勵函數
            nn.Linear(50, 1)  # 輸出層
        )

        device = "cpu"
        model = model.to(device)
        xTrainTensor = xTrainTensor.to(device)
        yTrainTensor = yTrainTensor.to(device)
        xDevTensor = xDevTensor.to(device)
        yDevTensor = yDevTensor.to(device)

        # 損失函數：使用MSELoss 與 學習函數：使用Adam
        lossfunc = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # ---------- 模型訓練 ----------
        epochs = 5001
        for epoch in range(epochs):
            model.train()  # 切換成訓練模式
            yPred = model(xTrainTensor)
            trainLoss = lossfunc(yPred, yTrainTensor)
            optimizer.zero_grad();trainLoss.backward();optimizer.step()
            if epoch % 400 == 0:
                with torch.no_grad():
                    model.eval()  # 切換成驗證模式
                    yDevPred = model(xDevTensor)  # 使用為訓練的資料進行損失函數驗證
                    validLoss = lossfunc(yDevPred, yDevTensor)
                # 可以注意到損失會不斷的下降
                print(f"Epoch:{epoch}, Train Loss:{trainLoss.item():.3f}, Valid Loss:{validLoss.item():.3f}")
                # 損失值小於81 會提前結束訓練
                if trainLoss.item() < 81:
                    break

        #

        return {}, {"model":model}

    @classmethod
    def M0_0_5(self, functionInfo):
        import os , copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from sklearn.metrics import accuracy_score
        import torch as torch
        import torch.nn as nn
        import torch.nn.functional as functional
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_5"])
        functionVersionInfo["Version"] = "M0_0_5"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        # 設定隨機種子
        torch.manual_seed(0)

        # 設定隨機種子
        xTrainTensor = globalObject['P0_0_5']["xTrainTensor"]
        yTrainTensor = globalObject['P0_0_5']["yTrainTensor"]
        xDevTensor = globalObject['P0_0_5']["xDevTensor"]
        yDevTensor = globalObject['P0_0_5']["yDevTensor"]

        trainDataset = TensorDataset(xTrainTensor, yTrainTensor)
        devDataset = TensorDataset(xDevTensor, yDevTensor)

        batch_size = 100
        trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

        x_, y_ = next(iter(trainLoader))
        print(x_.shape, y_.shape)                                           # torch.Size([100, 23]) torch.Size([100])

        # ========== MX_X_X ==========

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

        model = NeuralNetwork(xTrainTensor.shape[1])

        # 損失函數：使用NLLLoss 與 學習函數：使用Adam
        lossfunc = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 11
        # 收集所有 train 與 dev 的 Loss 與 Accs 值
        trainLossValueList, devLossValueList, trainAccsValueList, devAccsValueList = [], [], [], []

        for epoch in range(epochs):
            trainLossValue = 0
            trainAccValue = 0
            model.train()                                                           # 切換成訓練模式
            for xBatch, yBatch in trainLoader:                                      # 進行批次訓練
                pred = model(xBatch)
                trainLoss = lossfunc(pred, yBatch)
                optimizer.zero_grad();trainLoss.backward();optimizer.step()
                trainLossValue += trainLoss.item()                                  # 每一批都要進行損失值加總

                realTrainPred = torch.exp(pred)                                     # 將log_softmax輸出轉為softmax輸出
                topPred, topClass = realTrainPred.topk(1, dim=1)                    # 取出最大值及其索引
                trainAccValue += accuracy_score(yBatch, topClass)                   # 計算準確率

            # 驗證
            devLossValue = 0
            devAccValue = 0
            with torch.no_grad():
                model.eval()                                                        # 切換成驗證模式
                predDev = model(xDevTensor)
                devLoss = lossfunc(predDev, yDevTensor)

                realDevPred = torch.exp(predDev)                                    # 將log_softmax輸出轉為softmax輸出
                topPred, topClassDev = realDevPred.topk(1, dim=1)                   # 取出最大值及其索引
                devLossValue += devLoss.item()
                devAccValue += accuracy_score(yDevTensor, topClassDev) * 100

            trainLossValue = trainLossValue / len(trainLoader)
            trainAccValue = trainAccValue / len(trainLoader) * 100

            trainLossValueList.append(trainLossValue)
            devLossValueList.append(devLossValue)
            trainAccsValueList.append(trainAccValue)
            devAccsValueList.append(devAccValue)

            model.eval()
            if epoch % 1 == 0:
                print(f"Epoch:{epoch},TrainLoss:{trainLossValue:.3f},ValLoss:{devLossValue:.3f}" f"TrainAcc:{trainAccValue:.2f}%,ValAcc:{devAccValue:.2f}%")
        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_5"
        modelFilePath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_5/UCICreditCard.pt"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        torch.save(model.state_dict(), modelFilePath)

        return {}, {"ModelFilePath":modelFilePath}

    @classmethod
    def M0_0_6(self, functionInfo):
        import os, copy
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl

        torch.manual_seed(10)  # 設定隨機種子

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_6"])
        functionVersionInfo["Version"] = "M0_0_6"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        # 設定隨機種子
        xTrainTensor = globalObject['P0_0_6']["xTrainTensor"]
        yTrainTensor = globalObject['P0_0_6']["yTrainTensor"]

        class DataSet(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.nSample = len(x)

            def __getitem__(self, index):
                return self.x[index], self.y[index]

            def __len__(self):
                return self.nSample

        trainDataset = DataSet(xTrainTensor, yTrainTensor)

        trainLoader = DataLoader(dataset=trainDataset, batch_size=20, shuffle=True)

        # ========== MX_X_X ==========

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

        model = NeuralNetwork(xTrainTensor.shape[1])

        lossfunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        epochs = 200
        for i in range(epochs + 1):
            model.train()
            for xBatch, yBatch in trainLoader:
                pre = model(xBatch)
                labels = yBatch.view(-1)
                loss = lossfunc(pre, labels)
                optimizer.zero_grad();loss.backward();optimizer.step()
            if i % 50 == 0:
                print(f"Epoch:{i:3d}, Loss:{loss:.3f}")

        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_6"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        modelFilePath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_6/Iris.pt"
        torch.save(model.state_dict(), modelFilePath)

        return {}, {"ModelFilePath":modelFilePath}

    @classmethod
    def M0_0_7(self, functionInfo):
        import os,copy
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl

        torch.manual_seed(10)  # 設定隨機種子

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_7"])
        functionVersionInfo["Version"] = "M0_0_7"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        trainDataLoader = globalObject['P0_0_7']["TrainDataLoader"]

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

        device = torch.device('cpu')
        model = NeuralNetwork().to(device)

        lossfunc = nn.NLLLoss()
        optimizer = optim.Adadelta(model.parameters(), lr=0.5)

        def train(model, device, trainDataLoader, lossfunc, optimizer, epoch):
            model.train()
            for b_i, (x, y) in enumerate(trainDataLoader):
                x, y = x.to(device), y.to(device)
                predProb = model(x)
                loss = lossfunc(predProb, y)
                optimizer.zero_grad();loss.backward();optimizer.step()

                if b_i % 200 == 0:
                    num1 = b_i * len(x)
                    num2 = len(trainDataLoader.dataset)
                    num3 = 100 * b_i / len(trainDataLoader)
                    print('Epoch:{} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f}'.format(epoch, num1, num2, num3,loss.item()))

        epochs = 1
        for epoch in range(epochs):
            train(model, device, trainDataLoader, lossfunc, optimizer, epoch)
        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_7"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        modelFilePath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_7/MNIST.pt"
        torch.save(model.state_dict(), modelFilePath)

        return {}, {"ModelFilePath": modelFilePath}

    @classmethod
    def M0_0_9(self, functionInfo):
        import os,copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        import numpy
        import random
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import models
        from torch.optim.lr_scheduler import StepLR

        # 隨機種子
        numpy.random.seed(1234)
        random.seed(1234)
        torch.manual_seed(1234)

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_9"])
        functionVersionInfo["Version"] = "M0_0_9"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        trainDataSet = globalObject['P0_0_9']["TrainDataSet"]
        verifyDataSet = globalObject['P0_0_9']["VerifyDataSet"]
        trainDataLoader = globalObject['P0_0_9']["TrainDataLoader"]
        verifyDataLoader = globalObject['P0_0_9']["VerifyDataLoader"]

        # 載入預訓練模型
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # 保留 model 的輸入數，輸出改為2
        model.fc = nn.Linear(model.fc.in_features, 2)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # 損失函數
        lossfunc = nn.CrossEntropyLoss()
        # 學習器
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # 學習調度器 代表每step_size的步數學習率下降gamma optimizerScheduler.step() 為一步
        optimizerScheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        epochs = 1
        for epoch in range(epochs):
            model.train() # 模型訓練
            losses = 0.0
            corrects = 0
            for x , y in trainDataLoader:
                x , y = x.to(device) , y.to(device)
                outputs = model(x)
                loss = lossfunc(outputs, y)
                optimizer.zero_grad() ; loss.backward() ; optimizer.step()

                _ , preds = torch.max(outputs, 1) # 將輸出結果變成預測
                losses += loss.item() / x.size(0) # 加總損失函數
                corrects += torch.sum(preds == y.data) / x.size(0) # 正確的數量

            optimizerScheduler.step() # 一代執行一次學習調度器 下降學習函數
            trainLoss = losses / len(trainDataLoader) # 訓練損失函數加總
            trainAcc = corrects / len(trainDataLoader) # 訓練準確率加總

            model.eval() # 模型驗證
            losses = 0.0
            corrects = 0
            for x , y in verifyDataLoader:
                x , y = x.to(device) , y.to(device)
                outputs = model(x)
                loss = lossfunc(outputs, y)

                _, preds = torch.max(outputs, 1) # 將輸出結果變成預測
                losses += loss.item() / x.size(0) # 加總損失函數
                corrects += torch.sum(preds == y.data) / x.size(0)

            verifyLoss = losses / len(verifyDataLoader) # 驗證損失函數加總
            verifyAcc = corrects / len(verifyDataLoader) # 驗證準確率加總

            print(f"Epoch: {epoch}, Train Loss: {trainLoss:.4f}, Acc:{trainAcc:.4f}, Val Loss: {verifyLoss:.4f}, Acc:{verifyAcc:.4f}")

        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_9"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        torch.save(model.state_dict(), "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_9/model.pt")

        return {}, {}

    @classmethod
    def M0_0_10(self, functionInfo):
        import os , copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        import numpy
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from torch.utils.data.sampler import SubsetRandomSampler
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_10"])
        functionVersionInfo["Version"] = "M0_0_10"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        trainData = globalObject['P0_0_10']["TrainData"]

        devSize = 0.2
        idList = list(range(len(trainData)))
        numpy.random.shuffle(idList)                                        # 隨機切分驗證集
        splitSize = int(numpy.floor(devSize * len(trainData)))              # 切分數量
        trainIDList, devIDList = idList[splitSize:], idList[:splitSize]
        trainSampler = SubsetRandomSampler(trainIDList)                     # 訓練集
        devSampler = SubsetRandomSampler(devIDList)                         # 驗證集

        batchSize = 100
        trainDataLoader = DataLoader(trainData, batch_size=batchSize, sampler=trainSampler)
        devDataLoader = DataLoader(trainData, batch_size=batchSize, sampler=devSampler)
        print(len(trainDataLoader), len(devDataLoader))

        dataBatch, labelBatch = next(iter(trainDataLoader)) # 取得一批圖像 並分資料與標籤
        print(dataBatch.size(), labelBatch.size())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"deivce:{device}")

        import Example.P34PyTorch.package.cifar10_model as cifar10Model  # 匯入自訂模型

        epochs = 1  # 訓練代數
        endLoss = 0.65  # 結束時損失函數 可以提早結束
        model = cifar10Model.CNN().to(device)

        lossfunc = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainLossValueList = []
        devLossValueList = []
        xAxis = []
        for epoch in range(epochs + 1):
            trainLoss = 0
            model.train() # 訓練資料
            for x , y in tqdm(trainDataLoader): # 根據撈取來做進度條
                x , y = x.to(device) , y.to(device)
                pred = model(x)
                loss = lossfunc(pred, y) # 計算損失函數
                optimizer.zero_grad() ; loss.backward() ; optimizer.step()
                trainLoss += loss.item()
            trainLoss = trainLoss / len(trainDataLoader)
            xAxis.append(epoch)
            with torch.no_grad():
                devLoss = 0
                # 驗證資料
                model.eval()
                for x, y in tqdm(devDataLoader):
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = lossfunc(pred, y)  # 計算損失函數
                    devLoss += loss.item()
                devLoss = devLoss / len(devDataLoader)
            trainLossValueList.append(trainLoss)
            devLossValueList.append(devLoss)
            print(f"Epoch: {epoch}, TrainLoss: {trainLoss:.3f}, ValidLoss: {devLoss:.3f}")
            if trainLoss < endLoss:
                break

        plt.plot(xAxis, trainLossValueList, label="Training Loss")
        plt.plot(xAxis, devLossValueList, label="Validation Loss")
        plt.legend(frameon=False)
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.show()

        model = model.to("cpu")

        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_10"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        modelFile = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_10/cifar10_model.pt"  # 模型儲存位置
        torch.save(model.state_dict(), modelFile)

        return {}, {}

    @classmethod
    def M0_0_11(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        import numpy as np
        import torch
        from torch import nn, optim
        from torch.utils.data.sampler import SubsetRandomSampler
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_11"])
        functionVersionInfo["Version"] = "M0_0_11"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        trainData = globalObject['P0_0_11']["TrainData"]

        devSize = 0.2
        idList = list(range(len(trainData)))
        np.random.shuffle(idList)
        splitSize = int(np.floor(devSize * len(trainData)))
        trainIDList, devIDList = idList[splitSize:], idList[:splitSize]
        trainSampler = SubsetRandomSampler(trainIDList)
        devSampler = SubsetRandomSampler(devIDList)

        batch_size = 100
        trainDataLoader = DataLoader(trainData, batch_size=batch_size, sampler=trainSampler)
        devDataLoader = DataLoader(trainData, batch_size=batch_size, sampler=devSampler)
        print(len(trainDataLoader), len(devDataLoader))

        dataBatch, labelBatch = next(iter(trainDataLoader))
        print(dataBatch.size(), labelBatch.size())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Deivce:{device}")

        import Example.P34PyTorch.package.cifar10_resnet as cifar10Model
        modelFile = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_11/model.pt"
        epochs = 10
        endLoss = 0.45

        model = cifar10Model.CNN().to(device)

        lossfunc = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainLossValueList = []
        devLossValueList = []
        xAxis = []
        for epoch in range(epochs + 1):
            trainLoss = 0
            # 訓練資料
            model.train()
            for x, y in tqdm(trainDataLoader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = lossfunc(pred, y)  # 計算損失函數
                optimizer.zero_grad();loss.backward();optimizer.step()
                trainLoss += loss.item()
            trainLoss = trainLoss / len(trainDataLoader)

            xAxis.append(epoch)
            with torch.no_grad():
                devLoss = 0
                # 驗證資料
                model.eval()
                for x, y in tqdm(devDataLoader):
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = lossfunc(pred, y)  # 計算損失函數
                    devLoss += loss.item()
                devLoss = devLoss / len(devDataLoader)
            trainLossValueList.append(trainLoss)
            devLossValueList.append(devLoss)

            print(f"Epoch: {epoch}, Train Loss: {trainLoss:.3f}, Valid loss: {devLoss:.3f}")
            if trainLoss < endLoss:
                break

        plt.plot(xAxis, trainLossValueList, label="Training Loss")
        plt.plot(xAxis, devLossValueList, label="Validation Loss")
        plt.legend(frameon=False)
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.show()

        model = model.to("cpu")
        torch.save(model.state_dict(), modelFile)

        return {}, {}

    @classmethod
    def M0_0_12(self, functionInfo):
        import os,copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        import torch
        from torch import nn, optim

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_12"])
        functionVersionInfo["Version"] = "M0_0_11"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        xTrainTensor = globalObject['P0_0_12']["XTrainTensor"]
        yTrainTensor = globalObject['P0_0_12']["YTrainTensor"]

        torch.manual_seed(10) # 固定隨機種子

        class RNN(nn.Module):
            def __init__(self, inputSize, hiddenSize, numLayers):
                super().__init__()
                self.hiddenSize = hiddenSize
                self.rnn = nn.RNN(inputSize, hiddenSize, numLayers, batch_first=True)
                self.fc1 = nn.Linear(hiddenSize, 50)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(50, 1)

            def forward(self, x, hidden):
                out, hidden = self.rnn(x, hidden)
                out = out.view(-1, self.hiddenSize)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.fc2(out)
                return out, hidden

        model = RNN(51, 100, 1)
        lossfunc = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 10000
        losses = []
        for epoch in range(epochs + 1):
            pred , hidden = model(xTrainTensor, None)
            loss = lossfunc(pred , yTrainTensor)
            optimizer.zero_grad() ; loss.backward() ; optimizer.step()

            losses.append(loss.item())
            if epoch % 1000 == 0:
                print(f"Epoch:{epoch:5d}, Loss:{loss.item():.3f}")

        modelPath = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_12"
        os.makedirs(modelPath) if not os.path.isdir(modelPath) else None
        modelFile = "Example/P34PyTorch/file/result/V0_0_1/9999/M0_0_12/model.pt"
        torch.save(model.state_dict(), modelFile)

        return {}, {}

    @classmethod
    def M0_0_13(self, functionInfo):
        import numpy
        import pickle
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from string import punctuation
        from collections import Counter
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        torch.manual_seed(123)                                          # 固定隨機種子

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 資料前處理
        with open('common/common/file/data/text/imdb.pt', 'rb') as f:
            dataDict = pickle.load(f)                                   # 載入新聞資料

        reviewList = dataDict["review"]                                 # 標題資料
        labelList = dataDict["label"]                                   # 標籤資料
        reviewPreProssList = [review.lower() for review in reviewList]
        reviewPreProssList = [''.join([letter for letter in review if letter not in punctuation]) for review in reviewPreProssList] # string.punctuation : 所有的標點字元
        reviewsStr = ' '.join(reviewPreProssList)                                                   # 將所有的標題合併成一個字串
        reviewWordList = reviewsStr.split()                                                         # 將字串切割成單字
        countWords = Counter(reviewWordList)                                                        # 計算每個字出現的次數
        sortedReviewWords = countWords.most_common(len(reviewWordList))                             # 將字出現的次數由大到小排序
        vocabToToken = {word: index + 1 for index, (word, count) in enumerate(sortedReviewWords)}   # 將字與索引值做一個對應
        reviewsTokenized = []
        for review in reviewPreProssList:
            wordToToken = [vocabToToken[word] for word in review.split()]                           # 將每個字轉換成對應的token
            reviewsTokenized.append(wordToToken)                                                    # 將每個標題轉換成token
        encodedLabeList = [1 if label == 'pos' else 0 for label in labelList]                       # 將標籤轉換成數字
        reviewsLen = [len(review) for review in reviewsTokenized]                                   # 計算 reviews_tokenized 每則 review 的長度

        nZero = [index for index, wordlen in enumerate(reviewsLen) if wordlen == 0]                 # 找出 reviews_tokenized 每則 review 的長度為 0 的 index
                                                                                                    # 先做for 後作 in
        # encodedLabelList 轉成 numpy
        encodedLabelList = numpy.array([encodedLabeList[index] for index, wordlen in enumerate(reviewsLen) if wordlen > 0], dtype='float32')

        def makeFixedLengthMatrix(reviewsTokenized, number):
            fixedLengthMatrix = numpy.zeros((len(reviewsTokenized), number), dtype=int)             # 建立一個 len(reviewsTokenized) x number 的矩陣
            for index, review in enumerate(reviewsTokenized):
                reviewLen = len(review)                                                             # 計算每則 review 的長度
                if reviewLen <= number:                                                             # 如果每則 review 的長度小於 number
                    zeros = list(numpy.zeros(number - reviewLen))                                   # 建立一個長度為 200 - reviewLen 的 0 list
                    newList = zeros + review                                                        # 將 0 list 與 review 合併
                elif reviewLen > number:                                                            # 如果每則 review 的長度大於 200
                    newList = review[0:number]                                                      # 將每則 review 的長度取前 200 個字
                fixedLengthMatrix[index,:] = numpy.array(newList)                                   # 將每則 review 的長度為 200 的 index 填入 paddedReviews
            return fixedLengthMatrix

        reviewsFixedReviewsMatrix = makeFixedLengthMatrix(reviewsTokenized, 512)

        number = int(0.75 * len(reviewsFixedReviewsMatrix))
        xTrain = reviewsFixedReviewsMatrix[:number]
        yTrain = encodedLabelList[:number]
        xTest = reviewsFixedReviewsMatrix[number:]
        yTest = encodedLabelList[number:]

        xTrainTensor = torch.tensor(xTrain).to(device)
        yTrainTensor = torch.tensor(yTrain).to(device)
        xTestTensor = torch.tensor(xTest).to(device)
        yTestTensor = torch.tensor(yTest).to(device)

        trainDataset = TensorDataset(xTrainTensor, yTrainTensor)
        testDataset = TensorDataset(xTestTensor, yTestTensor)

        trainDataLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
        testDataLoader = DataLoader(testDataset, batch_size=32, shuffle=True)

        class LSTM(nn.Module):
            def __init__(self, nInput, nEmbed, nHidden, nOutput):
                super().__init__()
                self.embeddingLayer = nn.Embedding(nInput, nEmbed)
                self.lstmLayer = nn.LSTM(nEmbed, nHidden, num_layers=1)
                self.fcLayer = nn.Linear(nHidden, nOutput)

            def forward(self, x):
                x = self.embeddingLayer(x)
                out, hidden = self.lstmLayer(x)
                out = self.fcLayer(hidden[0].squeeze(0))
                return out

        nInput = len(vocabToToken) + 1
        model = LSTM(nInput, 100, 50, 1).to(device)

        lossfunc = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def accfunc (predictions, actual):
            roundedPredictions = torch.round(torch.sigmoid(predictions))
            success = (roundedPredictions == actual).float()
            accuracy = success.sum() / len(success)
            return accuracy

        epochs = 10
        for epoch in range(epochs):
            model.train()
            trainLoss , trainAcc ,testLoss ,testAcc= 0 ,0 ,0 ,0
            for x, y in tqdm(trainDataLoader):
                pred = model(x.T)
                pred = pred.squeeze()
                loss = lossfunc(pred, y)
                acc = accfunc(pred, y)
                optimizer.zero_grad() ; loss.backward() ; optimizer.step()
                trainLoss += loss.item()
                trainAcc += acc.item()

            with torch.no_grad():
                model.eval()
                for x, y in tqdm(testDataLoader):
                    pred = model(x.T).squeeze()
                    loss = lossfunc(pred, y)
                    acc = accfunc(pred, y)
                    testLoss += loss.item()
                    testAcc += acc.item()

            trainLoss = trainLoss / len(trainDataLoader)
            testLoss = testLoss / len(testDataLoader)
            trainAcc = trainAcc / len(trainDataLoader) * 100
            testAcc = testAcc / len(testDataLoader) * 100

            print(f"Epoch:{epoch:2d}, Train Loss: {trainLoss:.3f}, Val Loss:{testLoss:.3f}, Train Acc: {trainAcc:.2f}%, Test Acc:{testAcc:.2f}%")

        def sentimentInference(model, sentence):
            model.eval()
            sentence = sentence.lower()
            sentence = ''.join([c for c in sentence if c not in punctuation])
            tokenized = [vocabToToken[word] for word in sentence.split()]
            tokenized = numpy.pad(tokenized, (512 - len(tokenized), 0), 'constant')

            modelInput = torch.LongTensor(tokenized).to(device)
            modelInput = modelInput.unsqueeze(1)
            pred = torch.sigmoid(model(modelInput))
            pred = torch.round(pred, decimals=3)
            return pred.item()

        print("ddddd")
        out1 = sentimentInference(model, 'This film is horrible')
        print(f"{out1:.3f}")
        out2 = sentimentInference(model, 'Director tried too hard but this film is bad')
        print(f"{out2:.3f}")
        out3 = sentimentInference(model, 'Decent movie, although could be shorter')
        print(f"{out3:.3f}")
        out4 = sentimentInference(model, "I loved the movie, every part of it")
        print(f"{out4:.3f}")

        return {}, {}

    @classmethod
    def M0_0_14(self, functionInfo):
        # Agent             大腦
        # Environment       環境
        # state             狀態
        # action            動作
        # reward            獎勵
        # MDP (Markov Decision Process) 馬可夫決策過程
        # S: state 狀態
        # A: action 動作
        # P: transition probability 轉移機率
        # R: reward 獎勵
        import numpy as np

        R = np.array([                              # R Table 狀態與動作表 -1表示不可行動 0以上表示可行動(0,100)
            [-1, -1, -1, -1, 0, -1],                # state 0 (action1, action2, action3, action4, action5, action6)
            [-1, -1, -1, 0, -1, 100],               # state 1 (action1, action2, action3, action4, action5, action6)
            [-1, -1, -1, 0, -1, -1],                # state 2 (action1, action2, action3, action4, action5, action6)
            [-1, 0, 0, -1, 0, -1],                  # state 3 (action1, action2, action3, action4, action5, action6)
            [0, -1, -1, 0, -1, 100],                # state 4 (action1, action2, action3, action4, action5, action6)
            [-1, 0, -1, -1, 0, 100],                # state 5 (action1, action2, action3, action4, action5, action6)
        ], dtype='float')
        print(R)

        Q = np.zeros((6, 6))                                            # Q Table 建立一個跟R表一樣的全0 Matrix

        def findAvailableActions(state):
            currentState = R[state, :]                                  # 當前狀態與動作表
            availableAct = np.where(currentState >= 0)[0]               # 可用的行動
            return availableAct

        def getRandomAvailableAction(availableAct):
            action = int(np.random.choice(availableAct, size=1))        # 隨機選擇一個可用的行動
            return action

        def runQLearning(state, action, learn):                                 # Q學習(狀態, 動作, 衰減率)
            newState = action                                                   # 下一個狀態
            reward = R[state, action]                                           # 獲得獎勵
            Q[state, action] = reward + learn * np.max(Q[newState, :])          # 更新Q表
            done = True if newState == 5 else False                             # 結束條件
            score = (np.sum(Q) / np.max(Q) * 100) if (np.max(Q) > 0) else 0     # 獲得分數
            score = np.round(score, 2)
            return newState, reward, score, done                                # 下一個狀態, 獲得獎勵, 獲得分數, 結束條件

        epochs = 100                                                            # 訓練次數
        learn = 0.8                                                             # 學習率
        for epoch in range(epochs):
            state = np.random.randint(0, 6)                                     # 隨機選擇一個狀態
            for step in range(20):                                              # 每次訓練最多20步
                availableAct = findAvailableActions(state)                      # 找出可用的行動
                action = getRandomAvailableAction(availableAct)                 # 隨機選擇一個可用的行動
                newState, reward, score, done = runQLearning(state, action, learn)
                state = newState
                if done == True:
                    break
            print('epoch:', epoch, 'score:', score) if epoch % 10 == 0 else None
        Q = np.round(Q, 0)
        print(Q)

        state = 0                                                               # 開始狀態
        states = [state]                                                        # 紀錄每次訓練的狀態
        while state != 5:
            newState = np.argmax(Q[state, :])                                   # 選擇最大的Q值
            state = newState                                                    # 更新狀態
            states.append(newState)                                             # 紀錄每次訓練的狀態
        print(states)                                                           # 顯示最終的訓練結果

        return {}, {}

    @classmethod
    def M0_0_15(self, functionInfo):
        import numpy as np

        R = np.array([
            [-1, 0, 0, -1],     # state 0 (action1, action2, action3, action4)
            [-1, 0, -1, 0],     # state 1 (up, right, down, left)
            [-1, -1, 0, 0],     # state 2
            [0, 0, 0, -1],      # state 3
            [-1, -1, 0, 0],     # state 4
            [0, -1, -1, -1],    # state 5
            [0, -1, -1, -1],    # state 6
            [0, 100, -1, -1],   # state 7
            [-1, -1, -1, 0],    # state 8
        ], dtype='float')
        print(R)

        Q = np.zeros((9, 4))

        def findAvailableActions(state):
            currentState = R[state, :]                                  # 當前狀態與動作表
            availableAct = np.where(currentState >= 0)[0]               # 可用的行動
            return availableAct

        def getRandomAvailableAction(availableAct):
            action = int(np.random.choice(availableAct, size=1))        # 隨機選擇一個可用的行動
            return action

        def runQLearning(state, action, learn, decay):
            actionArr = [-3, 1, 3, -1]                                                  # up, right, down, left
            newState = state + actionArr[action]                                        # 下一個狀態
            reward = R[state, action]                                                   # 獲得獎勵
            maxValue = reward + learn * np.max(Q[newState, :])                          # 更新Q表
            Q[state, action] = Q[state, action] + decay * (maxValue - Q[state, action])
            done = True if newState == 8 else False                                     # 結束條件
            score = (np.sum(Q) / np.max(Q) * 100) if (np.max(Q) > 0) else 0             # 獲得分數
            score = np.round(score, 2)
            return newState, reward, done, score

        epochs = 30
        learn = 0.8  # 學習率
        decay = 0.9  # 衰減率
        for epoch in range(epochs):
            state = 0
            for step in range(20):
                availableAct = findAvailableActions(state)
                action = getRandomAvailableAction(availableAct)
                new_state, reward, done, score = runQLearning(state, action, learn, decay)
                state = new_state
                if done == True:
                    break
            print('epoch:', epoch, 'score:', score) if epoch % 10 == 0 else None
        Q = np.round(Q, 0)
        print(Q)

        state = 0
        states = [state]
        while state != 8:
            action = np.argmax(Q[state, :])
            actionArr = [-3, 1, 3, -1]
            newState = state + actionArr[action]
            state = newState
            states.append(state)
        print(states)

        return {}, {}

    @classmethod
    def M0_0_16(self, functionInfo):
        import numpy as np
        import gym

        np.random.seed(10)                                      # 重現性固定隨機種子

        gameEnv = gym.make('FrozenLake-v1', is_slippery=False)  # 遊戲環境

        actionSize = gameEnv.action_space.n                     # 行動數量
        stateSize = gameEnv.observation_space.n                 # 狀態數量

        Q = np.zeros((stateSize, actionSize))                   # 建立初始Q表

        greedy = 1                                              # 貪婪度
        epochs = 1000                                           # 訓練次數
        learn = 0.5                                             # 學習率
        decay = 0.9                                             # 衰減率
        for epoch in range(epochs):
            state = gameEnv.reset()[0]
            for step in range(50):
                if np.random.rand() > greedy:                   # 貪婪度
                    action = np.argmax(Q[state, :])             # 選擇最大的Q值
                else:
                    action = gameEnv.action_space.sample()      # 隨機選擇一個行動
                # 執行行動 newState, reward, terminated, truncated, info = step(action)
                # action: 行動
                # newState: 下一個狀態
                # reward: 獲得的獎勵
                # terminated: 當遊戲結束時，會回傳True
                # truncated: 當遊戲結束時，會回傳True
                # info: 遊戲的相關資訊
                newState, reward, terminated, truncated, info = gameEnv.step(action)            # 執行行動
                done = terminated or truncated                                                  # 結束條件
                maxValue = reward + decay * np.max(Q[newState, :])
                Q[state, action] = Q[state, action] + learn * (maxValue - Q[state, action])     # 更新Q表
                state = newState
                if done == True:
                    break
            score = np.sum(Q) / np.max(Q) * 100 if (np.max(Q) > 0) else 0
            score = np.round(score, 2)
            print('epoch:', epoch, 'score:', score) if epoch % 100 == 0 else None

            greedy = 0.01 + (0.09 * np.exp(0.005 * epoch))                                      # 根據代數更新貪婪程度

        Q = np.round(Q, 2)
        print(f"final Q:\n {Q}")

        state = gameEnv.reset()[0]
        states = [state]
        gameEnv.render()
        for step in range(50):                                                                  # 取得最佳動作
            action = np.argmax(Q[state, :])                                                     # 選擇最大的Q值
            newState, reward, terminated, truncated, info = gameEnv.step(action)                # 執行行動
            done = terminated or truncated                                                      # 結束條件
            state = newState                                                                    # 更新狀態
            states.append(state)                                                                # 紀錄狀態
            gameEnv.render()                                                                    # 顯示環境
            if done == True:
                break
        print(states)

        return {}, {}

    @classmethod
    def M0_0_17(self, functionInfo):
        import gym
        import numpy as np

        np.random.seed(10)  # 重現性固定隨機種子

        gameEnv = gym.envs.make('MountainCar-v0')                                       # 遊戲環境
        nState = gameEnv.observation_space.shape[0]                                     # 狀態值(非Type類是數值類)
        nAction = gameEnv.action_space.n                                                # 行動值(非Type類是數值類)

        numPosition = 10
        numSpeed = 10

        def digitizeState(observation):                                                 # 將狀態值離散化 observation: 狀態值
            def bins(clipMin, clipMax, num):                                            # 將狀態值分成num個區間
                return np.linspace(clipMin, clipMax, num + 1)[1:-1]

            carPosition, carSpeed = observation                                         # 獲得車子位置與速度
            digitized = [                                                               # 將狀態值分成numPosition個區間與numSpeed個區間
                np.digitize(carPosition, bins=bins(-1.2, 0.6, numPosition)),
                np.digitize(carSpeed, bins=bins(-0.07, 0.07, numSpeed)),
            ]
            return digitized[0] + (digitized[1] * numPosition)                          # 回傳離散化後的狀態值

        Q = np.random.uniform(low=-1, high=1, size=(numPosition * numSpeed, nAction))   # Q表格初始化 (10*10,3)

        epochs = 10000
        greedy = 1                                                                      # 貪婪度
        learn = 0.01                                                                    # 學習率
        decay = 0.99                                                                    # 衰減率
        greedyDecayRate = 0.998                                                         # 貪婪度衰減率
        totalScore = 0                                                                  # 總分數
        scores = []
        for epoch in range(epochs + 1):
            observation = gameEnv.reset()[0]
            state = digitizeState(observation)
            score = 0
            for step in range(300):
                if np.random.rand() > greedy:                                               # 貪婪度
                    action = np.argmax(Q[state, :])                                         # 選擇最大的Q值
                else:
                    action = gameEnv.action_space.sample()                                  # 隨機選擇一個行動
                newObservation, reward, terminated, truncated, info = gameEnv.step(action)  # 執行行動
                done = terminated or truncated                                              # 結束條件
                newState = digitizeState(newObservation)                                    # 離散化狀態值
                reward = -200 if done == True else reward                                   # 結束時獎勵-200

                position = newObservation[0]            # 車子位置
                if position >= 0.5:                     # 到達目標位置
                    reward += 2000
                elif position >= 0.45:                  # 靠近目標位置
                    reward += 100
                elif position >= 0.4:
                    reward += 20
                elif position >= 0.3:
                    reward += 10
                elif position >= 0.2:
                    reward += 5

                score = score + reward

                maxValue = reward + decay * np.max(Q[newState, :])
                Q[state, action] = Q[state, action] + learn * (maxValue - Q[state, action])
                state = newState
                if done == True:
                    break

            score = np.round(score, 2)
            totalScore = totalScore + score
            greedy = greedy * greedyDecayRate
            greedy = max(greedy, 0.01)

            if epoch % 100 == 0:
                print(f"Epoch:{epoch}, Total Score:{totalScore / 100}, Greedy:{greedy:.3f}")
                scores.append(totalScore / 100)
                totalScore = 0

            if epoch > 2000 and np.mean(scores[-20:]) > 1600:
                print("training completed!")
                break

        Q = np.round(Q, 2)

        observation = gameEnv.reset()[0]
        state = digitizeState(observation)
        states = [state]
        gameEnv.render()
        for step in range(300):
            action = np.argmax(Q[state, :])
            newObservation, reward, terminated, truncated, info = gameEnv.step(action)
            done = terminated or truncated
            newState = digitizeState(newObservation)

            observation = newObservation
            pos = observation[0]
            state = newState
            states.append(observation)
            gameEnv.render()

            if done == True:
                print(f"position: {pos:.3f}");
                print("Number of Steps", step + 1)
                break

        return {}, {}

    @classmethod
    def M0_0_18(self, functionInfo):
        import random
        import numpy as np
        import gym
        import torch
        from torch import nn
        from torch import optim
        import torch.nn.functional as F
        from collections import namedtuple, deque

        torch.manual_seed(10)                       # 重現性固定隨機種子
        random.seed(10)
        np.random.seed(10)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        gameEnv = gym.make('CartPole-v1')           # 遊戲環境
        gameEnv.reset()[0]                          # 遊戲初始化

        gamenNT = namedtuple('Tr', ('state', 'action', 'nextState', 'reward', 'done'))  # 建立命名元組

        class DQN(nn.Module):                                   # DQN模型
            def __init__(self, inputSize, outputSize):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(inputSize, 24)             # 全連接層
                self.fc2 = nn.Linear(24, 24)                    # 全連接層
                self.fc3 = nn.Linear(24, outputSize)            # 全連接層

            def forward(self, input):
                middle = F.relu(self.fc1(input))
                middle = F.relu(self.fc2(middle))
                output = self.fc3(middle)
                return output

        class ReplayMemory():                                   # 回放記憶體
            def __init__(self, capacity):
                self.capacity = capacity                        # 容量
                self.index = 0                                  # 索引
                self.memory = []                                # 記憶

            def pushMemory(self, state, action, nextState, reward, done):                   # 儲存記憶
                if len(self.memory) < self.capacity:                                        # 如果記憶體未滿
                    self.memory.append(None)                                                # 增加記憶體
                self.memory[self.index] = gamenNT(state, action, nextState, reward, done)   # 儲存記憶
                self.index = (self.index + 1) % self.capacity                               # 更新索引

            def sample(self, batchSize):                                                    # 隨機取樣
                return random.sample(self.memory, batchSize)                                # 回傳取樣結果

            def sampleTorch(self, batchSize):
                gamenNTs = self.sample(batchSize)  # 取樣
                stateBatch = np.vstack([tr.state for tr in gamenNTs if tr is not None])         # 狀態
                actionBatch = np.vstack([tr.action for tr in gamenNTs if tr is not None])       # 行為
                nextStateBatch = np.vstack([tr.nextState for tr in gamenNTs if tr is not None]) # 下一狀態
                rewardBatch = np.vstack([tr.reward for tr in gamenNTs if tr is not None])       # 獎勵
                doneBatch = np.vstack([tr.done for tr in gamenNTs if tr is not None])           # 結束

                states = torch.from_numpy(stateBatch).float().to(device)            # 轉換為張量
                actions = torch.from_numpy(actionBatch).long().to(device)           # 轉換為張量
                nextStates = torch.from_numpy(nextStateBatch).float().to(device)    # 轉換為張量
                rewards = torch.from_numpy(rewardBatch).float().to(device)          # 轉換為張量
                dones = torch.from_numpy(doneBatch).float().to(device)              # 轉換為張量

                return (states, actions, nextStates, rewards, dones)                # 回傳取樣結果

            def __len__(self):
                return len(self.memory)                                             # 回傳記憶長度

        class Agent:  # Agent
            def __init__(self, nState, nAction, device):
                self.seed = random.seed(10)                         # 隨機種子
                self.model = DQN(nState, nAction).to(device)        # DQN模型

                self.nState = nState                                # 狀態數量
                self.nAction = nAction                              # 行為數量

                self.memorySize = 2000                              # 記憶體大小
                self.memory = ReplayMemory(self.memorySize)         # 建立記憶體

                self.batchSize = 32                                                 # 批次大小
                self.gamma = 0.99                                                   # 折扣因子
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.0025)     # 學習優化器
                self.tStep = 0                                                      # 學習步數

            def step(self, state, action, nextState, reward, done):                 # 學習步驟
                self.memory.pushMemory(state, action, nextState, reward, done)      # 儲存記憶
                self.tStep = (self.tStep + 1) % 4                                   # 更新學習步數
                if self.tStep == 0:                                                 # 如果學習步數為0
                    if len(self.memory) > self.batchSize:                           # 如果記憶體大於批次大小
                        samples = self.memory.sampleTorch(self.batchSize)           # 取樣
                        self.learn(samples)                                         # 學習

            def learn(self, samples):
                states, actions, nextStates, rewards, dones = samples                   # 取樣結果
                qExpected = self.model(states).gather(1, actions)                       # 預期Q值
                qTargetsMax = self.model(nextStates).detach().max(1)[0].unsqueeze(1)
                qTargets = rewards + (self.gamma * qTargetsMax * (1 - dones))           # 目標Q值
                loss = F.mse_loss(qExpected, qTargets)                                  # 計算損失 (預期,實際)
                self.optimizer.zero_grad();loss.backward(); self.optimizer.step()       # 清空梯度

            def action(self, state, eps=0.):                                            # 行為
                if random.random() > eps:                                               # 如果隨機值大於閥值
                    state = torch.from_numpy(state).float().unsqueeze(0).to(device)     # 轉換為張量
                    self.model.eval()                                                   # 評估模式
                    with torch.no_grad():                                               # 不計算梯度
                        action_values = self.model(state)                               # 行為值
                    self.model.train()                                                  # 訓練模式
                    return np.argmax(action_values.cpu().data.numpy())                  # 回傳行為
                else:                                                                   # 如果隨機值小於閥值
                    return random.choice(np.arange(self.nAction))                       # 回傳行為

        nState = gameEnv.observation_space.shape[0]                         # 狀態數量
        nAction = gameEnv.action_space.n                                    # 行為數量

        agent = Agent(nState, nAction, device)                              # 建立Agent

        scores = []
        scoresWindow = deque(maxlen=100)
        epochs = 100                                                        # 學習週期
        maxT = 5000                                                         # 最大時間步數
        epsStart = 1.0                                                      # 開始閥值
        epsEnd = 0.001                                                      # 結束閥值
        epsDecay = 0.9995                                                   # 閥值衰減
        eps = epsStart

        for epoch in range(1, epochs + 1):
            state = gameEnv.reset()[0]
            score = 0
            for i in range(maxT):
                action = agent.action(state, eps)                                           # 行為
                nextState, reward, terminated, truncated, info = gameEnv.step(action)       # 執行行為
                done = terminated or truncated
                reward = reward if not done or score == 499 else -10                        # 設定獎勵
                agent.step(state, action, nextState, reward, done)                          # 學習步驟
                state = nextState
                score += reward
                if done:
                    break
            scoresWindow.append(score)
            scores.append(score)
            eps = max(epsEnd, epsDecay * eps)

            print('\rEpoch {:4} \tAverage Score: {:8.2f} \tReward: {:8.2f}'.format(epoch, np.mean(scoresWindow), score),end="")

            if epoch % 100 == 0:
                print('\rEpoch {:4} \tAverage Score: {:8.2f} \tEpsilon: {:8.3f}'.format(epoch, np.mean(scoresWindow),eps))

            if epoch > 10 and np.mean(scores[-10:]) > 450:
                break

        def playGame():
            state = gameEnv.reset()[0]                                                  # 重置環境
            epoch = 0                                                                   # 遊戲次數
            done = False                                                                # 結束標記
            while done == False:
                action = agent.action(state)
                nextState, reward, terminated, truncated, info = gameEnv.step(action)   # 執行行為
                done = terminated or truncated                                          # 結束條件
                gameEnv.render()                                                        # 繪製畫面
                state = nextState                                                       # 更新狀態
                epoch += 1                                                              # 更新步數
            print(f"done, epoch:{epoch}")

        playGame()

        return {}, {}

    @classmethod
    def M1_0_1(self, functionInfo):
        import os
        import copy
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M1_0_1"])
        functionVersionInfo["Version"] = "M1_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        trainData = globalObject[functionVersionInfo["DataVersion"]]["TrainData"]
        testData = globalObject[functionVersionInfo["DataVersion"]]["TestData"]

        # 神經網路模型
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                # 建立類神經網路各層
                self.flatten = nn.Flatten()  # 轉為一維向量
                # 繼承名稱無法修改
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(28 * 28, 512), # 輸入層 (輸入數量,輸出數量) 用Linear實現全連接層
                    nn.ReLU(), # 激勵函數 ReLU
                    # nn.MaxPool2d(2) # 池化層 此處用不到 但可以放在激勵函數後面
                    nn.Linear(512, 512), # 中間層 (輸入數量,輸出數量)
                    nn.ReLU(), # 激勵函數 ReLU
                    nn.Linear(512, 10) # 輸出層 (輸入數量,輸出數量)
                )

            def forward(self, x):
                # 定義資料如何通過類神經網路各層
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits

        # 訓練模型方法
        def runTrainModel(dataLoader, model, lossFN, optimizer):
            # 資料總筆數並將將模型設定為訓練模式(train)
            size = len(dataLoader.dataset)
            model.train()
            # 批次讀取資料進行訓練
            for batch, (x, y) in enumerate(dataLoader):
                # 將資料放置於 GPU 或 CPU
                x, y = x.to(device), y.to(device)
                pred = model(x)  # 計算預測值
                loss = lossFN(pred, y)  # 計算損失值（loss）
                optimizer.zero_grad()  # 重設參數梯度（gradient）
                loss.backward()  # 反向傳播（backpropagation）
                optimizer.step()  # 更新參數
                # 輸出訓練過程資訊
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(x)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # 測試模型方法
        def runTestModel(dataLoader, model, lossFN):
            # 資料總筆數
            size = len(dataLoader.dataset)
            # 批次數量
            numBatches = len(dataLoader)
            # 將模型設定為驗證模式
            model.eval()
            # 初始化數值
            testLoss, correct = 0, 0
            # 驗證模型準確度
            with torch.no_grad():  # 不要計算參數梯度
                for x, y in dataLoader:
                    # 將資料放置於 GPU 或 CPU
                    x, y = x.to(device), y.to(device)
                    # 計算預測值
                    pred = model(x)
                    # 計算損失值的加總值
                    testLoss += lossFN(pred, y).item()
                    # 計算預測正確數量的加總值
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 計算平均損失值與正確率
            testLoss = testLoss / numBatches
            correct = correct / size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")

        # 批次載入筆數：因為要分次處理與訓練所以要設定批次載入資料筆數
        batchSize = 64

        # 建立 DataLoader
        trainDataLoader = DataLoader(trainData, batch_size=batchSize)
        testDataLoader = DataLoader(testData, batch_size=batchSize)

        # 使用CPU(cpu)或是使用GPU(cuda)模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NeuralNetwork().to(device)
        print(f"Using {device} device")
        print(model)

        # 損失函數：使用交叉熵誤差CrossEntropy 與 學習函數：使用SGD 梯度下降
        lossFN = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # 訓練次數：設定同一批資料訓練的次數
        epochs = 5

        # 開始訓練模型
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            runTrainModel(trainDataLoader, model, lossFN, optimizer)
            runTestModel(testDataLoader, model, lossFN)
        print("模型訓練完成！")

        # 模型儲存位置
        pytorchModelPath = "Example/P34PyTorch/file/result/V1_0_1/9999/M1_0_1"
        pytorchModelFilePath = "Example/P34PyTorch/file/result/V1_0_1/9999/M1_0_1/Model.pth"

        # 儲存模型參數
        os.makedirs(pytorchModelPath) if not os.path.isdir(pytorchModelPath) else None
        torch.save(model.state_dict(), pytorchModelFilePath)
        # Accuracy: 65.6%, Avg loss: 1.069365
        return {}, {"PyTorchModelFilePath":pytorchModelFilePath}

