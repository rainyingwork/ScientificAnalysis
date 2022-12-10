
class ModelUse() :
    
    @classmethod
    def M0_0_1(self, functionInfo):
        import copy
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from torchvision.transforms import ToTensor, Lambda, Compose
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        trainData = globalObject[functionVersionInfo["DataVersion"]]["TrainData"]
        testData = globalObject[functionVersionInfo["DataVersion"]]["TestData"]

        # 神經網路模型
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

        # 訓練模型方法
        def runTrainModel(dataloader, model, loss_fn, optimizer):
            # 資料總筆數並將將模型設定為訓練模式(train)
            size = len(dataloader.dataset)
            model.train()
            # 批次讀取資料進行訓練
            for batch, (x, y) in enumerate(dataloader):
                # 將資料放置於 GPU 或 CPU
                x, y = x.to(device), y.to(device)
                pred = model(x)  # 計算預測值
                loss = loss_fn(pred, y)  # 計算損失值（loss）
                optimizer.zero_grad()  # 重設參數梯度（gradient）
                loss.backward()  # 反向傳播（backpropagation）
                optimizer.step()  # 更新參數
                # 輸出訓練過程資訊
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(x)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # 測試模型方法
        def runTestModel(dataloader, model, loss_fn):
            # 資料總筆數
            size = len(dataloader.dataset)
            # 批次數量
            num_batches = len(dataloader)
            # 將模型設定為驗證模式
            model.eval()
            # 初始化數值
            test_loss, correct = 0, 0
            # 驗證模型準確度
            with torch.no_grad():  # 不要計算參數梯度
                for X, y in dataloader:
                    # 將資料放置於 GPU 或 CPU
                    X, y = X.to(device), y.to(device)
                    # 計算預測值
                    pred = model(X)
                    # 計算損失值的加總值
                    test_loss += loss_fn(pred, y).item()
                    # 計算預測正確數量的加總值
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 計算平均損失值與正確率
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        # 批次載入資料筆數
        batch_size = 64

        # 建立 DataLoader
        trainDataLoader = DataLoader(trainData, batch_size=batch_size)
        testDataLoader = DataLoader(testData, batch_size=batch_size)

        # 使用CPU(cpu)或是使用GPU(cuda)模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NeuralNetwork().to(device)
        print(f"Using {device} device")
        print(model)

        # 損失函數lossFN 與 學習優化器optimizer
        lossFN = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # 設定 epochs 數
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

        return {}, {"PyTorchModelFilePath":pytorchModelFilePath}

