
class ModelUse() :
    
    @classmethod
    def M1_0_1(self, functionInfo):
        import copy
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from torchvision.transforms import ToTensor, Lambda, Compose
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
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

        # 損失函數：使用交叉熵誤差CrossEntropy 與 學習優化器optimizer
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
        pytorchModelFilePath = "UnitTest/PyTorch/file/result/V0_0_1/9999/M0_0_1/Model.pth"

        # 儲存模型參數
        torch.save(model.state_dict(), pytorchModelFilePath)
        # Accuracy: 65.6%, Avg loss: 1.069365
        return {}, {"PyTorchModelFilePath":pytorchModelFilePath}

    @classmethod
    def M0_0_1(self, functionInfo):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt

        torch.manual_seed(0)

        w = torch.tensor([1, 3, 5]).float()
        x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], dim=1)
        y = torch.mv(x, w) + torch.randn(100) * 0.3
        y = y.unsqueeze(1)
        print(x.shape, y.shape)

        model = nn.Sequential(nn.Linear(3, 1, bias=False))

        myloss = nn.MSELoss()
        myoptim = optim.Adam(model.parameters(), lr=0.1)

        losses = []
        epochs = 101
        for epoch in range(epochs):
            y_pred = model(x)
            loss = myloss(y_pred, y)
            losses.append(loss.item())
            myoptim.zero_grad()
            loss.backward()
            myoptim.step()

            if epoch % 20 == 0:
                print(f"epoch={epoch}, loss={loss.item():.3f}")

        # plt.plot(losses)
        # plt.xlabel("epoch")
        # plt.ylabel("losses")
        # plt.show()

        print(list(model.parameters()))

        return {}, {}