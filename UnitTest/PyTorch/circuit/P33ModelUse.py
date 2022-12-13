
class ModelUse() :

    @classmethod
    def M0_4_0(self, functionInfo):
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

    @classmethod
    def M0_5_0(self, functionInfo):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import torch
        import torch.nn as nn
        # 設定隨機種子
        torch.manual_seed(0)

        # 讀取資料
        data = pd.read_csv('UnitTest/PyTorch/file/data/YearPredictionMSD.csv', nrows=50000, header=None)
        print(data.head())
        print(data.shape) # (50000, 91)

        # 數據所有資料欄位
        cols = data.columns
        # 數據中所有為數值型態的資料欄位
        num_cols = data._get_numeric_data().columns
        print(list(set(cols) - set(num_cols))) # [] -> 代表所有都是數值欄位
        print(data.isnull().sum().sum()) # 0 -> 代表資料非常乾淨沒有空值

        outliers = []
        for i in range(data.shape[1]):
            min_t = data[data.columns[i]].mean() - (3 * data[data.columns[i]].std())
            max_t = data[data.columns[i]].mean() + (3 * data[data.columns[i]].std())
            count = 0
            for j in data[data.columns[i]]:
                if j < min_t or j > max_t:
                    count += 1
            percentage = count / data.shape[0]
            if percentage > 0.05:
                outliers.append(i)

        print(outliers) # []

        x = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        # 資料標準化
        x = (x - x.mean()) / x.std()

        print(x.head())

        # 拆分數據成2個子集，x_new : x_test = 80:20
        # 再拆分數據集x_new成2個子集, x_train : x_dev = 75:25
        x_new, x_test, y_new, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        x_train, x_dev, y_train, y_dev = train_test_split(x_new, y_new, test_size=0.25, random_state=0)
        print(x_train.shape, x_dev.shape, x_test.shape) # (30000, 90) (10000, 90) (10000, 90)


        x_train_torch = torch.tensor(x_train.values).float()
        y_train_torch = torch.tensor(y_train.values).float().unsqueeze(1) # 使用unsqueeze增加一個維度
        x_dev_torch = torch.tensor(x_dev.values).float()
        y_dev_torch = torch.tensor(y_dev.values).float().unsqueeze(1) # 使用unsqueeze增加一個維度
        x_test_torch = torch.tensor(x_test.values).float()
        y_test_torch = torch.tensor(y_test.values).float().unsqueeze(1) # 使用unsqueeze增加一個維度

        print(x_train_torch.shape, y_train_torch.shape) # torch.Size([30000, 90]) torch.Size([30000, 1])

        # 建立神經網路
        model = nn.Sequential(
            nn.Linear(x_train.shape[1], 200), # 輸入層
            nn.ReLU(), # 激勵函數
            nn.Linear(200, 50), # 中間層
            nn.ReLU(), # 激勵函數
            nn.Linear(50, 1) # 輸出層
        )

        device = "cpu"
        model = model.to(device)
        x_train_torch = x_train_torch.to(device)
        y_train_torch = y_train_torch.to(device)
        x_dev_torch = x_dev_torch.to(device)
        y_dev_torch = y_dev_torch.to(device)

        # 損失函數：使用MSELoss 與 學習函數：使用Adam
        myloss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # ---------- 模型訓練 ----------
        epochs = 5001
        for epoch in range(epochs):
            # 切換成訓練模式
            model.train()
            y_pred = model(x_train_torch)
            train_loss = myloss(y_pred, y_train_torch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (epoch % 400 == 0):
                with torch.no_grad():
                    # 切換成驗證模式
                    model.eval()
                    y_pred2 = model(x_dev_torch)
                    valid_loss = myloss(y_pred2, y_dev_torch)

                # 可以注意到損失會不斷的下降
                print(f"epoch={epoch},  train_loss:{train_loss.item():.3f},valid_loss:{valid_loss.item():.3f}")

                # 損失值小於81 會提前結束訓練
                if train_loss.item() < 81:
                    break

        # ---------- 模型驗證 ----------
        model = model.to("cpu")
        pred = model(x_test_torch)
        test_loss = myloss(pred, y_test_torch)
        print(f"test_loss: {test_loss.item():.3f}") # test_loss: 370.447
        for i in range(100, 110):
            print(f"truth:{y_test_torch[i].item():.0f}, pred:{pred[i].item():.0f}")

        return {}, {}

    @classmethod
    def M0_6_0(self, functionInfo):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        from sklearn.metrics import accuracy_score
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        # 設定隨機種子
        torch.manual_seed(10)
        np.random.seed(10)

        # 讀取資料
        data = pd.read_csv("UnitTest/PyTorch/file/data/UCI_Credit_Card.csv")
        print(data.shape) # (30000, 25)

        print(data.head())

        data = data.drop(columns=["ID"])

        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        print(x.shape, y.shape) # (30000, 23) (30000,)

        # 資料標準化
        x = (x - x.min()) / (x.max() - x.min())
        print(x.head())

        # 拆分數據成2個子集，x_new : x_test = 80:20
        # 再拆分數據集x_new成2個子集, x_train : x_dev = 75:25
        x_new, x_test, y_new, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        x_train, x_dev, y_train, y_dev = train_test_split(x_new, y_new, test_size=0.25, random_state=0)
        print(x_train.shape, x_dev.shape, x_test.shape) # (18000, 23) (6000, 23) (6000, 23)

        x_train_t = torch.tensor(x_train.values).float()
        y_train_t = torch.tensor(y_train.values).long()
        x_dev_t = torch.tensor(x_dev.values).float()
        y_dev_t = torch.tensor(y_dev.values).long()
        x_test_t = torch.tensor(x_test.values).float()
        y_test_t = torch.tensor(y_test.values).long()

        train_ds = TensorDataset(x_train_t, y_train_t)
        dev_ds = TensorDataset(x_dev_t, y_dev_t)
        test_ds = TensorDataset(x_test_t, y_test_t)

        batch_size = 100
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        x_, y_ = next(iter(train_loader))
        print(x_.shape, y_.shape) # torch.Size([100, 23]) torch.Size([100])

        class Classifier(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.hidden_1 = nn.Linear(input_size, 100)
                self.hidden_2 = nn.Linear(100, 100)
                self.output = nn.Linear(100, 2)

            def forward(self, x):
                z = F.relu(self.hidden_1(x))
                z = F.relu(self.hidden_2(z))
                o = F.log_softmax(self.output(z), dim=1)
                return o

        model = Classifier(x_train.shape[1])
        print(model)
        """
        Classifier(
          (hidden_1): Linear(in_features=23, out_features=100, bias=True)
          (hidden_2): Linear(in_features=100, out_features=100, bias=True)
          (output): Linear(in_features=100, out_features=2, bias=True)
        )
        """

        # 損失函數：使用NLLLoss 與 學習函數：使用Adam
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 51
        train_losses, dev_losses, train_accs, dev_accs = [], [], [], []

        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0

            model.train()
            for x_batch, y_batch in train_loader:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                ps = torch.exp(pred)
                top_p, top_class = ps.topk(1, dim=1)
                train_acc += accuracy_score(y_batch, top_class)

            # 驗證
            dev_loss = 0
            dev_acc = 0

            with torch.no_grad():
                model.eval()
                pred_dev = model(x_dev_t)
                dev_loss = criterion(pred_dev, y_dev_t)

                ps_dev = torch.exp(pred_dev)
                top_p, top_class_dev = ps_dev.topk(1, dim=1)

                dev_loss = dev_loss.item()
                dev_acc = accuracy_score(y_dev_t, top_class_dev) * 100

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader) * 100

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            train_accs.append(train_acc)
            dev_accs.append(dev_acc)

            model.eval()
            if epoch % 10 == 0:
                print(f"Epoch:{epoch}, train_loss:{train_loss:.3f}, val_loss:{dev_loss:.3f}" \
                      f"\t  train_acc:{train_acc:.2f}%, val_acc:{dev_acc:.2f}%")

        # fig = plt.figure(figsize=(15, 5))
        # plt.plot(train_losses, label="training loss")
        # plt.plot(dev_losses, label="validation loss", linestyle="--")
        # plt.legend(frameon=False, fontsize=15)
        # plt.show()

        # fig = plt.figure(figsize=(15, 5))
        # plt.plot(train_accs, label="training accuracy")
        # plt.plot(dev_accs, label="validation accuracy", linestyle="--")
        # plt.legend(frameon=False, fontsize=15)
        # plt.show()

        torch.save(model.state_dict(), "UnitTest/PyTorch/file/result/V0_6_0/9999/M0_6_0/credit_model.pt")

        model2 = Classifier(x_test.shape[1])
        model2.load_state_dict(torch.load("UnitTest/PyTorch/file/result/V0_6_0/9999/M0_6_0/credit_model.pt"))

        model2.eval()
        test_pred = model2(x_test_t)
        test_pred = torch.exp(test_pred)
        value, y_test_pred = test_pred.topk(1, dim=1)
        acc_test = accuracy_score(y_test_t, y_test_pred) * 100
        print(f"acc_test:{acc_test:.2f}%") # acc_test:81.63%

        return {}, {}


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
        pytorchModelFilePath = "UnitTest/PyTorch/file/result/V0_0_1/9999/M0_0_1/Model.pth"

        # 儲存模型參數
        torch.save(model.state_dict(), pytorchModelFilePath)
        # Accuracy: 65.6%, Avg loss: 1.069365
        return {}, {"PyTorchModelFilePath":pytorchModelFilePath}

