class ModelUse():

    @classmethod
    def M0_0_1(self, functionInfo):
        import torch
        import numpy as np
        import glob # 多圖撈取模組
        from PIL import Image

        # torch.Tensor 浮點數張量，可用於GPU計算
        # torch.tensor 一般張量，視輸入而定

        example01 = torch.tensor(1,dtype=torch.int16)
        print(example01.shape) # torch.Size([])
        print(example01) # tensor(1,dtype=torch.int16)

        example02 = torch.Tensor([1, 2, 3, 4, 5])
        print(example02.shape) # torch.Size([5])
        print(example02) # tensor([1., 2., 3., 4., 5.]) 沒有dtype代表就是float，也會看到後面有.
        print(example02[:3]) # tensor([1., 2., 3.]) 操作的方式其實與一般操作很像
        print(example02[:-1]) # tensor([1., 2., 3., 4.])

        example03 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        print(example03.shape) # torch.Size([2, 3])
        print(example03) # tensor([[1., 2., 3.],[4., 5., 6.]])

        example04 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        print(example04)  # tensor([[1., 2., 3.],[4., 5., 6.]])
        print(example04.tolist())  # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 轉成純陣列
        print(example04.numel())  # 6 張量中元素的總數

        print(torch.arange(1, 6, 2)) # tensor([1,3,5]) torch.arange(start,end,step) 整數切
        print(torch.linspace(1, 10, 3)) # tensor([1.0,5.5,10.0]) torch.linspace(start,end,step) 均勻切
        print(torch.randn(2, 3)) # tensor([[2.05,-0.02,-0.17],[0.23,-0.28,-0.33]]) 為浮點數的X維張量
        print(torch.randperm(5)) # tensor([4, 2, 1, 3, 0]) 為整數的X維張量
        print(torch.eye(2, 3)) # tensor([[1., 0., 0.],[0., 1., 0.]]) 對角線為1的X維張量

        example05 = torch.arange(0, 6)
        print(example05) # tensor([0, 1, 2, 3, 4, 5]) torch.arange(start,end) 整數切
        print(example05.view(2, 3)) # tensor([[0, 1, 2],[3, 4, 5]]) 切成2個3列陣列
        print(example05.view(-1, 2)) # tensor([[0, 1],[2, 3],[4, 5]]) 切成n個2列陣列，n由系統運算

        example06 = example05.view(2, 3)
        example06 = example06.unsqueeze(1) # 在軸1擴展維度
        print(example06) # tensor([[[0, 1, 2]],[[3, 4, 5]]])
        print(example06.shape) # torch.Size([2, 1, 3])

        example07 = torch.arange(0, 6)
        example07 = example07.view(1, 1, 1, 2, 3)
        print(example07.shape) # torch.Size([1, 1, 1, 2, 3])

        example08 = example07.squeeze(0) # 降低軸0維度
        print(example08.shape) # torch.Size([1, 1, 2, 3])
        example09 = example08.squeeze() # 刪除維度為1的軸
        print(example09.shape) # torch.Size([2, 3])

        example10 = torch.arange(0, 12)
        print(example10) # tensor([0,1,2,3,4,5,6,7,8,9,10,11])
        print(example10.resize_(2,6)) # tensor([[0,1,2,3,4,5],[6,7,8,9,10,11]])
        print(example10.resize_(1,6)) # tensor([0,1,2,3,4,5])
        print(example10.resize_(3,6)) # tensor([[0,1,2,3,4,5],[6,7,8,9,10,11],[4322,4550,43228,4322,432,432]]) 多出來的張量會給數字

        # 讀取單一圖檔
        pandaNP = np.array(Image.open('Example/P36PyTorch/file/data/imgs/panda1.jpg'))
        pandaTensor = torch.from_numpy(pandaNP)
        print(pandaTensor.shape) # torch.Size([426, 640, 3])

        # 讀取多個圖檔
        pandaList = glob.glob("Example/P36PyTorch/file/data/imgs/*.jpg")
        print(pandaList)
        pandaNPList = []
        for panda in pandaList:
             tempPanda = Image.open(panda).resize((224, 224))
             pandaNPList.append(np.array(tempPanda))

        # 轉成多維向量
        pandaNPList = np.array(pandaNPList)
        pandaTensor = torch.from_numpy(pandaNPList)
        print(pandaTensor.shape) # torch.Size([4, 224, 224, 3])

        return {}, {}

    @classmethod
    def M0_0_2(self, functionInfo):
        import torch

        torch.manual_seed(0) # 設定隨機種子

        # ========== RX_X_X ==========

        w = torch.tensor([1, 3, 5]).float() # 等同於 torch.Tensor([1, 3, 5])
        x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], dim=1)  # torch.cat 連接張量(需確認一下相關方式) torch.ones 為全1的X維張量 , torch.randn 為浮點數的X維張量
        # torch.mm() 是正常的矩陣相乘，(a,b) * (b,c) = (a,c)
        # torch.mv() 是矩陣與向量相乘，類似torch.mm()是(a,b) * (b,1) = (a,1)
        # torch.mul() 是矩陣的點乘，即對應的位相乘，會要求shape一樣,返回也是一樣大小的矩陣
        # torch.dot() 類似torch.mul()，但是是向量的對應位相乘在求和，返回一個tensor值
        y = torch.mv(x, w) + torch.randn(100) * 0.3
        print(x.shape, y.shape) # torch.Size([100, 3]) torch.Size([100])

        # ========== MX_X_X ==========

        # 先隨機給三個數字，經由梯度下降算出正確的數字
        # 為了能夠進行反向傳播計算，必須將該物件的 requires_grad 方法設置為 True
        wPred = torch.randn(3, requires_grad=True)
        # 損失函數值，該值越高學習越快，該值越低學習越慢，但該值越高會越容易跳過局部最優
        lr = 0.01

        lossValueList = [] # 損失值紀錄
        epochs = 200 # 重複計算次數
        for epoch in range(epochs + 1):
            yPred = torch.mv(x, wPred)
            loss = torch.mean((y - yPred) ** 2)  # 計算MSE
            wPred.grad = None  # 清除上一次計算的梯度值
            loss.backward() # loss 向輸入側進行反向傳播，這時候 w_pred.grad 就會有值
            wPred.data = wPred.data - lr * wPred.grad.data  # 梯度下降更新 原本資料 - 損失函數 * 差異梯度
            lossValueList.append(loss.item())  # 記錄該次的loss值
            if (epoch) % 50 == 0 : # 印出50倍數代的相關數據
                print(f"epoch:{epoch}, Loss: {loss.item():.3f}")

        # 印出損失值過程
        print(lossValueList)
        # 印出最終預設的值
        print(wPred.data)
        return {}, {}

    @classmethod
    def M0_0_3(self, functionInfo):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt

        torch.manual_seed(0)  # 設定隨機種子

        # ========== RX_X_X ========== 相關說明可以參考 M0_0_2

        w = torch.tensor([1, 3, 5]).float()
        x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], dim=1)
        y = torch.mv(x, w) + torch.randn(100) * 0.3
        y = y.unsqueeze(1)  # 在軸1擴展維度
        print(x.shape, y.shape)

        # ========== MX_X_X ==========
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
            optimizer.zero_grad()  # 清除上一次計算的梯度值
            loss.backward()  # loss 向輸入側進行反向傳播
            optimizer.step()  # 做梯度下降更新
            if epoch % 20 == 0:  # 印出20倍數代的相關數據
                print(f"epoch={epoch}, loss={loss.item():.3f}")

        # 印出損失值過程
        print(lossList)
        # 印出最終預設的值
        print(list(model.parameters()))

        return {}, {}

    @classmethod
    def M0_0_4(self, functionInfo):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import torch
        import torch.nn as nn
        # 設定隨機種子
        torch.manual_seed(0)

        # 讀取資料
        data = pd.read_csv('Example/P36PyTorch/file/data/YearPredictionMSD.csv', nrows=50000, header=None)
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
    def M0_0_5(self, functionInfo):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
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
        data = pd.read_csv("Example/P36PyTorch/file/data/UCI_Credit_Card.csv")
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

        torch.save(model.state_dict(), "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_5/credit_model.pt")

        model2 = Classifier(x_test.shape[1])
        model2.load_state_dict(torch.load("Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_5/credit_model.pt"))

        model2.eval()
        test_pred = model2(x_test_t)
        test_pred = torch.exp(test_pred)
        value, y_test_pred = test_pred.topk(1, dim=1)
        acc_test = accuracy_score(y_test_t, y_test_pred) * 100
        print(f"acc_test:{acc_test:.2f}%") # acc_test:81.63%

        return {}, {}

    @classmethod
    def M0_0_6(self, functionInfo):
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import pandas as pd

        torch.manual_seed(10)

        iris_data = pd.read_csv('Example/P36PyTorch/file/data/iris.csv')
        print(iris_data.head())

        print(iris_data['Species'].unique())

        labelencoder = LabelEncoder()
        iris_data['Species'] = labelencoder.fit_transform(iris_data['Species'])
        print(iris_data['Species'].unique())

        iris_x = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        iris_y = iris_data['Species']
        print(iris_x.shape, iris_y.shape)

        iris_x = (iris_x - iris_x.min()) / (iris_x.max() - iris_x.min())
        print(iris_x.head())

        train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.15, random_state=10)
        print(train_x.shape, test_x.shape)

        train_x_t = torch.tensor(train_x.values).float()
        train_y_t = torch.tensor(train_y.values).long().unsqueeze(1)
        test_x_t = torch.tensor(test_x.values).float()
        test_y_t = torch.tensor(test_y.values).long().unsqueeze(1)

        print(train_x_t.shape, train_y_t.shape)

        class dataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.n_sample = len(x)

            def __getitem__(self, index):
                return self.x[index], self.y[index]

            def __len__(self):
                return self.n_sample

        train_ds = dataset(train_x_t, train_y_t)

        train_loader = DataLoader(dataset=train_ds, batch_size=20, shuffle=True)

        class Iris_Model(nn.Module):
            def __init__(self):
                super(Iris_Model, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(train_x.shape[1], 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 3)
                )

            def forward(self, x):
                return self.net(x)

        model = Iris_Model()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        epochs = 200
        for i in range(epochs + 1):
            # model.train()
            for samples, labels in train_loader:
                pre = model(samples)
                labels = labels.view(-1)
                loss = criterion(pre, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i % 50 == 0:
                print(f"epch={i:3d}, loss={loss:.3f}")

        torch.save(model.state_dict(), "iris_model.pt")

        model2 = Iris_Model()
        model2.load_state_dict(torch.load("Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_6/iris_model.pt"))

        pred = model2(test_x_t)
        _, pred_class = torch.max(pred, dim=1)
        n_correct = (test_y_t.view(-1) == pred_class).sum()
        print(f"valid_acc={n_correct / len(test_x_t):.4f}")

        return {}, {}

    @classmethod
    def M0_0_7(self, functionInfo):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import matplotlib.pyplot as plt

        torch.manual_seed(0)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_data = datasets.MNIST('Example/P36PyTorch/file/data/', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('Example/P36PyTorch/file/data/', train=False, transform=transform)

        train_x = train_data.data
        train_y = train_data.targets
        test_x = test_data.data
        test_y = test_data.targets
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        img, label = train_data[0]
        print(img.shape)  # shape:(C,H,W)

        print(img.min(), img.max())

        img2 = img.permute(1, 2, 0)  # shape:(H,W,C)
        plt.imshow(img2, cmap='gray')

        print(label)

        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=500, shuffle=False)

        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                self.cn1 = nn.Conv2d(1, 16, 3, 1)
                self.cn2 = nn.Conv2d(16, 32, 3, 1)
                self.dp1 = nn.Dropout(0.10)
                self.dp2 = nn.Dropout(0.25)
                self.fc1 = nn.Linear(12 * 12 * 32, 64)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.cn1(x)
                x = F.relu(x)
                x = self.cn2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = self.dp1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = self.dp2(x)
                op = F.log_softmax(x, dim=1)
                return op

        device = torch.device('cpu')
        model = ConvNet().to(device)

        myloss = nn.NLLLoss()
        myoptim = optim.Adadelta(model.parameters(), lr=0.5)

        def train(model, device, train_dataloader, myloss, myoptim, epoch):
            model.train()

            for b_i, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                pred_prob = model(X)
                loss = myloss(pred_prob, y)

                myoptim.zero_grad()
                loss.backward()
                myoptim.step()

                if b_i % 200 == 0:
                    num1 = b_i * len(X)
                    num2 = len(train_dataloader.dataset)
                    num3 = 100 * b_i / len(train_dataloader)
                    print('epoch:{} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                        epoch, num1, num2, num3, loss.item()))

        epochs = 3
        for epoch in range(epochs):
            train(model, device, train_dataloader, myloss, myoptim, epoch)

        def test(model, device, test_dataloader, myloss):
            model.eval()
            loss = 0
            success = 0
            with torch.no_grad():
                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    pred_prob = model(X)
                    loss += myloss(pred_prob, y).item()

                    pred = pred_prob.argmax(dim=1, keepdim=True)
                    success += pred.eq(y.view_as(pred)).sum().item()

                    num1 = loss / len(test_dataloader)
                    num2 = len(test_dataloader.dataset)
                    num3 = 100 * success / len(test_dataloader.dataset)
                    print('Overall Loss: {:.4f}, Overall Accuracy: {}/{} ({:.2f}%)'.format(
                        num1, success, num2, num3))

        torch.save(model.state_dict(), "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_7/mnist_model.pt")

        model2 = ConvNet()
        model2.load_state_dict(torch.load("Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_7/mnist_model.pt"))
        model2 = model2.to(device)

        test(model2, device, test_dataloader, myloss)

        model2 = model2.to("cpu")
        sample_data, sample_targets = next(iter(test_dataloader))
        print(sample_data.shape)

        print(sample_data[10].shape)

        plt.imshow(sample_data[10][0], cmap='gray')

        pred_label = model2(sample_data).max(dim=1)[1][10]
        print(pred_label)

        print(f"Model prediction is : {pred_label}")
        print(f"Ground truth is : {sample_targets[10]}")

        return {}, {}

    @classmethod
    def M0_0_8(self, functionInfo):
        from PIL import Image
        import torch
        from torchvision import transforms
        from torchvision import models
        import numpy as np
        import pandas as pd

        img = Image.open("Example/P36PyTorch/file/data/dog.jpg")
        img.show()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor()
        ])

        img2 = preprocess(img)
        print(img2.shape)

        img3 = torch.unsqueeze(img2, 0)
        print(img3.shape)

        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        resnet.eval()
        out = resnet(img3)

        out_numpy = out.detach().numpy()  # 轉為NumPy
        out_class = np.argmax(out_numpy, axis=1)  # 找出最大值的索引
        print(out_class)

        df = pd.read_csv("Example/P36PyTorch/file/data/imagenet_classes.csv", header=None)
        print(df.head())

        label = df.iloc[out_class].values
        print(label)

        score = torch.nn.functional.softmax(out, dim=1)[0] * 100
        print(score.shape)

        print(f"score:{score[out_class].item():.2f}")

        return {}, {}

    @classmethod
    def M0_0_9(self, functionInfo):

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import random
        import torchvision
        from torchvision import datasets, models
        from torchvision import transforms

        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        train_dataset = datasets.ImageFolder(
            root="Example/P36PyTorch/file/data/bees_ants/train",
            transform=train_transforms
        )
        val_dataset = datasets.ImageFolder(
            root="Example/P36PyTorch/file/data/bees_ants/val",
            transform=val_transforms
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        print(model.fc)
        model.fc = nn.Linear(model.fc.in_features, 2)
        print(model.fc)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        from torch.optim.lr_scheduler import StepLR
        exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        epochs = 21
        for epoch in range(epochs):
            model.train()
            losses = 0.0
            corrects = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                losses += loss.item() / inputs.size(0)
                corrects += torch.sum(preds == labels.data) / inputs.size(0)

            exp_lr_scheduler.step()
            train_loss = losses / len(train_loader)
            train_acc = corrects / len(train_loader)

            model.eval()
            losses = 0.0
            corrects = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                losses += loss.item() / inputs.size(0)
                corrects += torch.sum(preds == labels.data) / inputs.size(0)

            val_loss = losses / len(val_loader)
            val_acc = corrects.double() / len(val_loader)

            print(f"epoch: {epoch}, Train loss: {train_loss:.4f}, acc:{train_acc:.4f}, \
            Val loss: {val_loss:.4f}, acc:{val_acc:.4f}")

        inputs, classes = next(iter(val_loader))  # 取得一批圖像
        class_name = val_dataset.classes

        outputs = model(inputs.to(device))  # 預測輸出
        _, preds = torch.max(outputs, 1)
        title = [class_name[x] for x in preds]
        print(inputs.shape)
        print(title)

        import matplotlib.pyplot as plt

        out = torchvision.utils.make_grid(inputs)  # 顯示圖像
        out = out.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        out = std * out + mean
        out = np.clip(out, 0, 1)

        plt.imshow(out)
        if title is not None:
            plt.title(title)

        plt.show()

        torch.save(model.state_dict(), "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_9/bee.pt")

        return {}, {}

    @classmethod
    def M0_0_10Train(self, functionInfo):
        import numpy as np
        import random
        import torch
        from torch import nn, optim
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torch.utils.data.sampler import SubsetRandomSampler
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        from tqdm import tqdm  # pip install tqdm

        import Example.P36PyTorch.package.cifar10_model as cifar10_model
        model_file = "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_10/cifar10_model.pt"
        epochs = 50
        end_loss = 0.65

        torch.manual_seed(10)
        np.random.seed(10)
        random.seed(10)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
        ])

        train_data = datasets.CIFAR10('Example/P36PyTorch/file/data/cifar10/train', train=True, download=True, transform=transform)
        print(train_data.data.shape)

        dev_size = 0.2
        idx = list(range(len(train_data)))
        np.random.shuffle(idx)
        split_size = int(np.floor(dev_size * len(train_data)))
        train_idx, dev_idx = idx[split_size:], idx[:split_size]
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)

        batch_size = 100
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        dev_loader = DataLoader(train_data, batch_size=batch_size, sampler=dev_sampler)
        print(len(train_loader), len(dev_loader))

        data_batch, label_batch = next(iter(train_loader))
        print(data_batch.size(), label_batch.size())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"deivce:{device}")

        model = cifar10_model.CNN().to(device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        dev_losses = []
        x_axis = []
        for epoch in range(epochs + 1):
            train_loss = 0
            # 訓練資料
            model.train()
            for data, target in tqdm(train_loader):
                data = data.to(device)
                target = target.to(device)

                pred = model(data)
                loss = loss_function(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            x_axis.append(epoch)
            with torch.no_grad():
                dev_loss = 0
                # 驗證資料
                model.eval()
                for data_dev, target_dev in tqdm(dev_loader):
                    data_dev = data_dev.to(device)
                    target_dev = target_dev.to(device)

                    dev_pred = model(data_dev)
                    loss2 = loss_function(dev_pred, target_dev)
                    dev_loss += loss2.item()

                dev_loss = dev_loss / len(dev_loader)

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            print(f"epoch: {epoch}, Train_loss: {train_loss:.3f}, Valid_loss: {dev_loss:.3f}")

            if train_loss < end_loss:
                break

        plt.plot(x_axis, train_losses, label="training loss")
        plt.plot(x_axis, dev_losses, label="validation loss")
        plt.legend(frameon=False)
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.show()

        model = model.to("cpu")
        torch.save(model.state_dict(), model_file)

        return {}, {}

    @classmethod
    def M0_0_10Test(self, functionInfo):
        import torch
        from torch import nn
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        import Example.P36PyTorch.package.cifar10_model as cifar10_model
        model_file = "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_10/cifar10_model.pt"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
        ])

        test_data = datasets.CIFAR10('Example/P36PyTorch/file/data/cifar10/test', train=False, download=True, transform=transform)

        batch_size = 100
        test_loader = DataLoader(test_data, batch_size=batch_size)
        print(len(test_loader))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"deivce:{device}")

        model = cifar10_model.CNN()
        model.load_state_dict(torch.load(model_file))
        # print(model)

        model = model.to(device)

        loss_function = nn.NLLLoss()

        num_correct = 0.0
        test_loss = 0
        # 測試資料
        model.eval()
        for data_test, target_test in test_loader:
            data_test = data_test.to(device)
            target_test = target_test.to(device)

            test_pred = model(data_test)
            loss3 = loss_function(test_pred, target_test)
            test_loss += loss3.item()
            _, predicted = torch.max(test_pred, 1)
            num_correct += (predicted == target_test).float().sum()

        test_loss = test_loss / len(test_loader)
        num_correct = num_correct / (len(test_loader) * batch_size)
        print(f"test_loss: {test_loss:.3f}, correct: {num_correct:.3f}")

        return {}, {}

    @classmethod
    def M0_0_11Train(self, functionInfo):
        import numpy as np
        import random
        import torch
        from torch import nn, optim
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torch.utils.data.sampler import SubsetRandomSampler
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        import Example.P36PyTorch.package.cifar10_resnet as cifar10_model
        model_file = "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_11/cifar10_resnet.pt"
        epochs = 10
        end_loss = 0.45

        torch.manual_seed(10)
        np.random.seed(10)
        random.seed(10)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
        ])

        train_data = datasets.CIFAR10('Example/P36PyTorch/file/data/cifar10/train', train=True, download=True, transform=transform)
        print(train_data.data.shape)

        dev_size = 0.2
        idx = list(range(len(train_data)))
        np.random.shuffle(idx)
        split_size = int(np.floor(dev_size * len(train_data)))
        train_idx, dev_idx = idx[split_size:], idx[:split_size]
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)

        batch_size = 100
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        dev_loader = DataLoader(train_data, batch_size=batch_size, sampler=dev_sampler)
        print(len(train_loader), len(dev_loader))

        data_batch, label_batch = next(iter(train_loader))
        print(data_batch.size(), label_batch.size())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"deivce:{device}")

        model = cifar10_model.CNN().to(device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        dev_losses = []
        x_axis = []
        for epoch in range(epochs + 1):
            train_loss = 0
            # 訓練資料
            model.train()
            for data, target in tqdm(train_loader):
                data = data.to(device)
                target = target.to(device)

                pred = model(data)
                loss = loss_function(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            x_axis.append(epoch)
            with torch.no_grad():
                dev_loss = 0
                # 驗證資料
                model.eval()
                for data_dev, target_dev in tqdm(dev_loader):
                    data_dev = data_dev.to(device)
                    target_dev = target_dev.to(device)

                    dev_pred = model(data_dev)
                    loss2 = loss_function(dev_pred, target_dev)
                    dev_loss += loss2.item()

                dev_loss = dev_loss / len(dev_loader)

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            print(f"epoch: {epoch}, Train_loss: {train_loss:.3f}, Valid_loss: {dev_loss:.3f}")

            if train_loss < end_loss:
                break

        plt.plot(x_axis, train_losses, label="training loss")
        plt.plot(x_axis, dev_losses, label="validation loss")
        plt.legend(frameon=False)
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.show()

        model = model.to("cpu")
        torch.save(model.state_dict(), model_file)

        return {}, {}

    @classmethod
    def M0_0_11Test(self, functionInfo):
        import torch
        from torch import nn
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        import Example.P36PyTorch.package.cifar10_resnet as cifar10_model
        model_file = "Example/P36PyTorch/file/result/V0_0_1/9999/M0_0_11/cifar10_resnet.pt"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616))
        ])

        test_data = datasets.CIFAR10('Example/P36PyTorch/file/data/cifar10/test', train=False, download=True, transform=transform)

        batch_size = 100
        test_loader = DataLoader(test_data, batch_size=batch_size)
        print(len(test_loader))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"deivce:{device}")

        model = cifar10_model.CNN()
        model.load_state_dict(torch.load(model_file))
        # print(model)

        model = model.to(device)

        loss_function = nn.NLLLoss()

        num_correct = 0.0
        test_loss = 0
        # 測試資料
        model.eval()
        for data_test, target_test in test_loader:
            data_test = data_test.to(device)
            target_test = target_test.to(device)

            test_pred = model(data_test)
            loss3 = loss_function(test_pred, target_test)
            test_loss += loss3.item()
            _, predicted = torch.max(test_pred, 1)
            num_correct += (predicted == target_test).float().sum()

        test_loss = test_loss / len(test_loader)
        num_correct = num_correct / (len(test_loader) * batch_size)
        print(f"test_loss: {test_loss:.3f}, correct: {num_correct:.3f}")

        return {}, {}

    @classmethod
    def M0_0_12(self, functionInfo):
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn, optim
        from sklearn.model_selection import train_test_split

        torch.manual_seed(10)

        df = pd.read_csv('Example/P36PyTorch/file/data/Sales_Transactions_dataset_weekly.csv')
        df.head()

        df = df.iloc[:, 1:53]
        print(df.shape)

        plot_data = df.sample(5, random_state=0)
        x = range(1, 53)
        plt.figure(figsize=(10, 5))
        for i, row in plot_data.iterrows():
            plt.plot(x, row)
        plt.legend(plot_data.index)
        plt.xlabel("Weeks")
        plt.ylabel("Sales")
        plt.show()

        x = df.iloc[:, :-1]  # w0 - w50
        y = df.iloc[:, -1]  # w51

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        x_train_t = torch.tensor(x_train.values).float().unsqueeze(1)  # (batch, seq, input)
        y_train_t = torch.tensor(y_train.values).float().unsqueeze(1)
        x_test_t = torch.tensor(x_test.values).float().unsqueeze(1)
        y_test_t = torch.tensor(y_test.values).float().unsqueeze(1)
        print(x_train_t.shape, y_train_t.shape)
        print(x_test_t.shape, y_test_t.shape)

        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.hidden_size = hidden_size
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
                self.fc1 = nn.Linear(hidden_size, 50)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(50, 1)

            def forward(self, x, hidden):
                out, hidden = self.rnn(x, hidden)
                out = out.view(-1, self.hidden_size)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.fc2(out)
                return out, hidden

        model = RNN(51, 100, 1)
        print(model)

        myloss = nn.MSELoss()
        myoptim = optim.Adam(model.parameters(), lr=0.001)

        epochs = 10000
        losses = []
        for i in range(epochs + 1):
            pred, hidden = model(x_train_t, None)
            loss = myloss(y_train_t, pred)

            myoptim.zero_grad()
            loss.backward()
            myoptim.step()

            losses.append(loss.item())
            if i % 1000 == 0:
                print(f"epoch:{i:5d}, loss:{loss.item():.3f}")

        x_range = range(len(losses))
        plt.xlabel('epochs')
        plt.ylabel('loss function')
        plt.plot(x_range, losses)
        plt.show()

        pred, hidden = model(x_test_t, None)
        loss = myloss(y_test_t, pred)
        print(f"loss:{loss.item():.3f}")

        for i in range(20):
            truth = y_test_t[i].item()
            pred2 = pred[i].item()
            print(f"truth:{truth:3.0f}   pred:{pred2:5.2f}")

        return {}, {}

    @classmethod
    def M0_0_13(self, functionInfo):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(123)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        import os
        from tqdm import tqdm
        import pickle

        def read_data():
            review_list = []
            label_list = []
            for label in ['pos', 'neg']:
                for fname in tqdm(os.listdir(f"Example/P36PyTorch/file/data/aclImdb/train/{label}/")):
                    if 'txt' not in fname:
                        continue
                    with open(os.path.join(f"Example/P36PyTorch/file/data/aclImdb/train/{label}/", fname), encoding="utf-8") as f:
                        review_list += [f.read()]
                        label_list += [label]

            # 使用 pickle 儲存
            mydict = {'review': review_list, 'label': label_list}
            with open('Example/P36PyTorch/file/data/imdb.pt', 'wb') as f:
                pickle.dump(mydict, f)

        with open('Example/P36PyTorch/file/data/imdb.pt', 'rb') as f:
            new_dict = pickle.load(f)
        review_list = new_dict["review"]
        label_list = new_dict["label"]

        print(len(review_list), len(label_list))

        print(review_list[0])

        review_list2 = [review.lower() for review in review_list]
        print(review_list2[0])

        from string import punctuation
        # string.punctuation : 所有的標點字元

        review_list3 = [''.join([letter for letter in review if letter not in punctuation]) for review in review_list2]

        print(review_list3[0])

        reviews_blob = ' '.join(review_list3)

        review_words = reviews_blob.split()
        print(review_words[:10])

        from collections import Counter
        count_words = Counter(review_words)
        print(count_words['bromwell'])

        sorted_review_words = count_words.most_common(len(review_words))
        print(sorted_review_words[:10])  # 印出前10名出現最多的單詞

        vocab_to_token = {word: idx + 1 for idx, (word, count) in enumerate(sorted_review_words)}
        print(list(vocab_to_token.items())[:10])  # 印出字典前10個元素

        reviews_tokenized = []
        for review in review_list3:
            word_to_token = [vocab_to_token[word] for word in review.split()]
            reviews_tokenized.append(word_to_token)
        print(review_list3[0])
        print("len=", len(review_list3[0]))
        print()
        print(reviews_tokenized[0])
        print("len=", len(reviews_tokenized[0]))

        encoded_label_list = [1 if label == 'pos' else 0 for label in label_list]

        reviews_len = [len(review) for review in reviews_tokenized]  # 計算 reviews_tokenized 每則 review 的長度
        print(reviews_len[:10])

        n_zero = [i for i, n in enumerate(reviews_len) if n == 0]
        print(n_zero)  # 沒有長度為0的分詞 review

        import numpy as np
        # encoded_label_list 轉成 numpy
        encoded_label_list = np.array([encoded_label_list[i] for i, n in enumerate(reviews_len) if n > 0],
                                      dtype='float32')

        def pad_sequence(reviews_tokenized, num):
            padded_reviews = np.zeros((len(reviews_tokenized), num), dtype=int)
            for idx, review in enumerate(reviews_tokenized):
                review_len = len(review)
                if review_len <= num:
                    zeros = list(np.zeros(num - review_len))
                    new_sequence = zeros + review
                elif review_len > num:
                    new_sequence = review[0:num]
                padded_reviews[idx, :] = np.array(new_sequence)
            return padded_reviews

        padded_reviews = pad_sequence(reviews_tokenized, 512)

        num = int(0.75 * len(padded_reviews))
        x_train = padded_reviews[:num]
        y_train = encoded_label_list[:num]
        x_val = padded_reviews[num:]
        y_val = encoded_label_list[num:]

        x_train_t = torch.tensor(x_train).to(device)
        y_train_t = torch.tensor(y_train).to(device)
        x_val_t = torch.tensor(x_val).to(device)
        y_val_t = torch.tensor(y_val).to(device)
        print(x_train_t.shape)  # (batch_size, input_dim)
        print(x_val_t.shape)

        train_ds = TensorDataset(x_train_t, y_train_t)
        val_ds = TensorDataset(x_val_t, y_val_t)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)

        x_, y_ = next(iter(train_loader))
        print(x_.shape, y_.shape)

        class LSTM(nn.Module):
            def __init__(self, n_input, n_embed, n_hidden, n_output):
                super().__init__()
                self.n_hidden = n_hidden
                self.embedding_layer = nn.Embedding(n_input, n_embed)
                self.lstm_layer = nn.LSTM(n_embed, n_hidden, num_layers=1)
                self.fc_layer = nn.Linear(n_hidden, n_output)

            def forward(self, x):
                # x shape: (seq,batch)=(512,32)
                x = self.embedding_layer(x)
                # x shape: (seq,batch,feature)=(512,32,100)
                out, hidden = self.lstm_layer(x)
                # hidden[0] shape: (num_layers, batch, feature)=(1,32,50)
                out = self.fc_layer(hidden[0].squeeze(0))
                return out

        n_input = len(vocab_to_token) + 1
        model = LSTM(n_input, 100, 50, 1).to(device)
        print(model)
        print(n_input)

        myloss = nn.BCEWithLogitsLoss()
        myoptim = optim.Adam(model.parameters(), lr=0.001)

        def myacc(predictions, ground_truth):
            rounded_predictions = torch.round(torch.sigmoid(predictions))
            success = (rounded_predictions == ground_truth).float()  # convert into float for division
            accuracy = success.sum() / len(success)
            return accuracy

        # import time
        # 訓練
        epochs = 10
        for epoch in range(epochs):
            # time_start=time.time()
            train_loss = 0
            val_loss = 0
            train_acc = 0
            val_acc = 0

            model.train()
            for xx, yy in (train_loader):
                pred = model(xx.T)
                pred = pred.squeeze()

                loss = myloss(pred, yy)
                myoptim.zero_grad()
                loss.backward()
                myoptim.step()

                acc = myacc(pred, yy)

                train_loss += loss.item()
                train_acc += acc.item()

            with torch.no_grad():
                model.eval()
                for xx2, yy2 in (val_loader):
                    pred2 = model(xx2.T).squeeze()
                    loss2 = myloss(pred2, yy2)
                    acc2 = myacc(pred2, yy2)

                    val_loss += loss2.item()
                    val_acc += acc2.item()

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = train_acc / len(train_loader) * 100
            val_acc = val_acc / len(val_loader) * 100

            print(f"epoch:{epoch:2d}, train_loss: {train_loss:.3f}, val_loss:{val_loss:.3f}, " \
                  f"train_acc: {train_acc:.2f}%, val_acc:{val_acc:.2f}%")

        def sentiment_inference(model, sentence):
            model.eval()

            # text transformations
            sentence = sentence.lower()
            sentence = ''.join([c for c in sentence if c not in punctuation])
            tokenized = [vocab_to_token[word] for word in sentence.split()]
            tokenized = np.pad(tokenized, (512 - len(tokenized), 0), 'constant')

            # model inference
            model_input = torch.LongTensor(tokenized).to(device)
            model_input = model_input.unsqueeze(1)
            pred = torch.sigmoid(model(model_input))
            pred2 = torch.round(pred, decimals=3)
            return pred.item()

        out1 = sentiment_inference(model, 'This film is horrible')
        print(f"{out1:.3f}")
        out2 = sentiment_inference(model, 'Director tried too hard but this film is bad')
        print(f"{out2:.3f}")
        out3 = sentiment_inference(model, 'Decent movie, although could be shorter')
        print(f"{out3:.3f}")
        out4 = sentiment_inference(model, "I loved the movie, every part of it")
        print(f"{out4:.3f}")

        return {}, {}

    @classmethod
    def M0_0_14(self, functionInfo):

        import numpy as np
        import matplotlib.pyplot as plt

        R = np.array([
            [-1, -1, -1, -1, 0, -1],  # state 0
            [-1, -1, -1, 0, -1, 100],  # state 1
            [-1, -1, -1, 0, -1, -1],  # state 2
            [-1, 0, 0, -1, 0, -1],  # state 3
            [0, -1, -1, 0, -1, 100],  # state 4
            [-1, 0, -1, -1, 0, 100]  # state 5
        ], dtype='float')
        print(f"R:\n {R}")

        Q = np.zeros((6, 6))
        print(f"Q:\n {Q}")

        gamma = 0.8
        state = 1

        def available_actions(state):
            current_state = R[state, :]
            av_act = np.where(current_state >= 0)[0]
            return av_act

        av_act = available_actions(state)
        print(f"av_act: {av_act}")

        def get_action(av_act):
            action = int(np.random.choice(av_act, size=1))
            return action

        action = get_action(av_act)
        print(f"action: {action}")

        def Q_learning(state, action, gamma):
            new_state = action
            reward = R[state, action]
            Q[state, action] = reward + gamma * np.max(Q[new_state, :])
            if new_state == 5:
                done = True
            else:
                done = False

            if (np.max(Q) > 0):
                score = np.sum(Q) / np.max(Q) * 100
            else:
                score = 0
            score = np.round(score, 2)

            return new_state, reward, done, score

        new_state, reward, done, score = Q_learning(state, action, gamma)
        print(f"new_state:{new_state},reward:{reward},done:{done}, Q:\n {Q}")

        epochs = 600
        scores = []
        for epoch in range(epochs):
            state = np.random.randint(0, 6)
            for step in range(20):
                av_act = available_actions(state)
                action = get_action(av_act)
                new_state, reward, done, score = Q_learning(state, action, gamma)
                state = new_state
                if done:
                    # print(f"done. score:{score}")
                    break
            scores.append(score)

        Q = np.round(Q, 0)
        print(f"final Q:\n {Q}")

        plt.plot(scores)
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.show()

        states = []
        state = 0
        states.append(state)

        while state != 5:
            new_state = np.argmax(Q[state, :])
            state = new_state
            states.append(state)

        print(states)

        return {}, {}

    @classmethod
    def M0_0_15(self, functionInfo):
        import numpy as np
        import matplotlib.pyplot as plt

        R = np.array([
            [-1, 0, 0, -1],  # state 0
            [-1, 0, -1, 0],  # state 1
            [-1, -1, 0, 0],  # state 2
            [0, 0, 0, -1],  # state 3
            [-1, -1, 0, 0],  # state 4
            [0, -1, -1, -1],  # state 5
            [0, -1, -1, -1],  # state 6
            [0, 100, -1, -1],  # state 7
            [-1, -1, -1, 0],  # state 8
        ], dtype='float')

        Q = np.zeros((9, 4))

        gamma = 0.9
        alpha = 0.8
        state = 0

        def available_actions(state):
            current_state = R[state, :]
            av_act = np.where(current_state >= 0)[0]
            return av_act

        av_act = available_actions(state)
        print(f"av_act: {av_act}")

        def get_action(av_act):
            action = int(np.random.choice(av_act, size=1))
            return action

        action = get_action(av_act)
        print(f"action: {action}")

        def Q_learning(state, action, gamma):
            if action == 0:  # up
                new_state = state - 3
            if action == 1:  # right
                new_state = state + 1
            if action == 2:  # down
                new_state = state + 3
            if action == 3:  # left
                new_state = state - 1

            reward = R[state, action]
            max_value = reward + gamma * np.max(Q[new_state, :])
            Q[state, action] = Q[state, action] + alpha * (max_value - Q[state, action])
            if new_state == 8:
                done = True
            else:
                done = False

            if (np.max(Q) > 0):
                score = np.sum(Q) / np.max(Q) * 100
            else:
                score = 0
            score = np.round(score, 2)

            return new_state, reward, done, score

        new_state, reward, done, score = Q_learning(state, action, gamma)
        print(f"new_state:{new_state},reward:{reward},done:{done}, Q:\n {Q}")

        epochs = 30
        scores = []
        for epoch in range(epochs):
            state = 0
            for step in range(20):
                av_act = available_actions(state)
                action = get_action(av_act)
                new_state, reward, done, score = Q_learning(state, action, gamma)
                state = new_state
                if done:
                    # print(f"done. score:{score}")
                    break
            scores.append(score)

        Q = np.round(Q, 0)
        print(f"final Q:\n {Q}")

        plt.plot(scores)
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.show()

        states = []
        state = 0
        states.append(state)

        while state != 8:
            action = np.argmax(Q[state, :])
            if action == 0:  # up
                new_state = state - 3
            if action == 1:  # right
                new_state = state + 1
            if action == 2:  # down
                new_state = state + 3
            if action == 3:  # left
                new_state = state - 1
            state = new_state
            # print(state)
            states.append(state)

        print(states)

        return {}, {}

    @classmethod
    def M0_0_16(self, functionInfo):
        import numpy as np
        import gym  # pip install gym==0.23.1
        import matplotlib.pyplot as plt

        np.random.seed(10)

        env = gym.make('FrozenLake-v1', is_slippery=False)

        action_size = env.action_space.n
        state_size = env.observation_space.n
        print(action_size, state_size)

        qtable = np.zeros((state_size, action_size))

        eps = 1
        scores = []
        gamma = 0.9
        alpha = 0.5

        epochs = 3000
        for epoch in range(epochs):
            state = env.reset()[0]
            score = 0

            for step in range(50):
                if np.random.rand() > eps:
                    action = np.argmax(qtable[state, :])
                else:
                    action = env.action_space.sample()

                # new_state, reward, done, info = env.step(action)
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                max_value = reward + gamma * np.max(qtable[new_state, :])
                qtable[state, action] += alpha * (max_value - qtable[state, action])
                state = new_state

                if done:
                    break

            if (np.max(qtable) > 0):
                score = np.sum(qtable) / np.max(qtable) * 100
            else:
                score = 0
            score = np.round(score, 2)

            scores.append(score)

            eps = 0.01 + (0.09 * np.exp(0.005 * epoch))

        Q = np.round(qtable, 2)
        print(f"final Q:\n {Q}")

        plt.plot(scores)
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.show()

        # !pip install pygame
        states = []

        state = env.reset()[0]
        env.render()
        states.append(state)

        step = 0
        done = False
        # 最多執行50步
        for step in range(50):
            # 取得最佳動作
            action = np.argmax(qtable[state, :])
            # print("action:",action)

            # 執行動作
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = new_state
            states.append(state)
            env.render()

            if done:
                print("Number of Steps", step + 1)
                break

        print(states)

        return {}, {}

    @classmethod
    def M0_0_17(self, functionInfo):
        import gym
        import numpy as np
        import matplotlib.pyplot as plt

        # 忽略 warning
        import warnings
        warnings.filterwarnings("ignore")

        np.random.seed(10)

        env = gym.envs.make('MountainCar-v0')
        n_state = env.observation_space.shape[0]  # car position, car velocity
        n_action = env.action_space.n  # acc to left, no acc, acc to right
        print(n_state, n_action)

        # 計算轉換成離散值的臨界值
        def bins(clip_min, clip_max, num):
            return np.linspace(clip_min, clip_max, num + 1)[1:-1]

        car_position = bins(-1.2, 0.6, 10)  # 分成10份
        car_velocity = bins(-0.07, 0.07, 10)  # 分成10份
        print(car_position)
        print(car_velocity)

        # 將連續值轉換成離散變數
        num_pos = 10
        num_v = 10

        def digitize_state(observation):
            car_pos, car_v = observation
            digitized = [
                np.digitize(car_pos, bins=bins(-1.2, 0.6, num_pos)),
                np.digitize(car_v, bins=bins(-0.07, 0.07, num_v)), ]

            return digitized[0] + (digitized[1] * num_pos)

        # Q表格初始化
        # qtable=np.zeros((20*14, n_action))
        qtable = np.random.uniform(low=-1, high=1, size=(num_pos * num_v, n_action))
        print(qtable.shape)

        # 設定變數
        eps = 1
        scores = []
        gamma = 0.99
        alpha = 0.01
        eps_decay_rate = 0.998

        epochs = 10000
        tot_score = 0
        scores = []
        for epoch in range(epochs + 1):
            observation = env.reset()[0]  # get car_pos, car_v
            state = digitize_state(observation)  # get state
            score = 0

            for step in range(300):
                rad = np.random.rand()
                # print(rad, eps)
                if rad > eps:
                    action = np.argmax(qtable[state, :])
                else:
                    action = env.action_space.sample()
                    # action=np.random.choice(n_action)

                new_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                new_state = digitize_state(new_observation)  # get new state
                # print(new_state)

                if done:
                    reward = -200

                pos = new_observation[0]  # car position
                if pos >= 0.5:
                    reward += 2000
                elif pos >= 0.45:
                    reward += 100
                elif pos >= 0.4:
                    reward += 20
                elif pos >= 0.3:
                    reward += 10
                elif pos >= 0.2:
                    reward += 5

                score += reward

                max_value = reward + gamma * np.max(qtable[new_state, :])
                qtable[state, action] += alpha * (max_value - qtable[state, action])

                # observateion=new_observation

                state = new_state

                if done:
                    break

            score = np.round(score, 2)
            # scores.append(score)
            tot_score += score
            eps = eps * eps_decay_rate
            eps = max(eps, 0.01)

            if epoch % 100 == 0:
                print(f"epoch:{epoch},score:{tot_score / 100}, eps={eps:.3f}")
                scores.append(tot_score / 100)
                tot_score = 0

            if epoch > 2000 and np.mean(scores[-20:]) > 1600:
                print("training completed!")
                break

        Q = np.round(qtable, 2)

        plt.plot(scores)
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.show()

        states = []

        observation = env.reset()[0]
        state = digitize_state(observation)

        env.render()
        states.append(observation)

        step = 0
        done = False

        # 最多執行300步
        for step in range(300):
            # 取得最佳動作
            action = np.argmax(qtable[state, :])

            # 執行動作
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            new_state = digitize_state(new_observation)

            observation = new_observation
            pos = observation[0]
            state = new_state
            states.append(observation)
            env.render()

            if done:
                print(f"position: {pos:.3f}")
                print("Number of Steps", step + 1)
                break

        # print(states)

        return {}, {}

    @classmethod
    def M0_0_18(self, functionInfo):
        import numpy as np
        import matplotlib.pyplot as plt
        import gym
        import random
        import torch
        from torch import nn
        from torch import optim
        import torch.nn.functional as F
        from collections import namedtuple, deque

        torch.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        env = gym.make('CartPole-v1')

        env.reset()[0]
        n_state = env.observation_space.shape[0]
        n_action = env.action_space.n
        print(n_state, n_action)

        state = env.reset()[0]
        print(state)
        state_size = env.observation_space.shape[0]
        # state = np.reshape(state, [1, state_size])
        # print(state)

        Tr = namedtuple('Tr', ('state', 'action', 'next_state', 'reward', 'done'))

        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(DQN, self).__init__()

                self.fc1 = nn.Linear(state_size, 24)
                self.fc2 = nn.Linear(24, 24)
                self.fc3 = nn.Linear(24, action_size)

            def forward(self, state):
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        batch_size = 32
        capacity = 10000

        class ReplayMemory:
            def __init__(self, capacity):
                self.capacity = capacity
                self.memory = []
                self.index = 0

            def push(self, state, action, next_state, reward, done):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.index] = Tr(state, action, next_state, reward, done)
                self.index = (self.index + 1) % self.capacity

            def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

            def sample_torch(self, batch_size):
                Trs = self.sample(batch_size)
                state_batch = np.vstack([tr.state for tr in Trs if tr is not None])
                action_batch = np.vstack([tr.action for tr in Trs if tr is not None])
                next_state_batch = np.vstack([tr.next_state for tr in Trs if tr is not None])
                reward_batch = np.vstack([tr.reward for tr in Trs if tr is not None])
                done_batch = np.vstack([tr.done for tr in Trs if tr is not None])

                states = torch.from_numpy(state_batch).float().to(device)
                actions = torch.from_numpy(action_batch).long().to(device)
                next_states = torch.from_numpy(next_state_batch).float().to(device)
                rewards = torch.from_numpy(reward_batch).float().to(device)
                dones = torch.from_numpy(done_batch).float().to(device)

                return (states, actions, next_states, rewards, dones)

            def __len__(self):
                return len(self.memory)

        class Agent:
            def __init__(self, n_state, n_action):
                self.n_state = n_state
                self.n_action = n_action
                self.seed = random.seed(10)
                self.buffer_size = 2000
                self.batch_size = 32
                self.gamma = 0.99

                self.model = DQN(n_state, n_action).to(device)
                self.memory = ReplayMemory(self.buffer_size)

                self.optimizer = optim.Adam(self.model.parameters(), lr=0.0025)
                self.t_step = 0

            def step(self, state, action, next_state, reward, done):
                self.memory.push(state, action, next_state, reward, done)
                self.t_step = (self.t_step + 1) % 4
                if self.t_step == 0:
                    if len(self.memory) > self.batch_size:
                        samples = self.memory.sample_torch(self.batch_size)
                        self.learn(samples)

            def learn(self, samples):
                states, actions, next_states, rewards, dones = samples
                q_expected = self.model(states).gather(1, actions)
                q_targets_max = self.model(next_states).detach().max(1)[0].unsqueeze(1)
                q_targets = rewards + (self.gamma * q_targets_max * (1 - dones))
                loss = F.mse_loss(q_expected, q_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            def act(self, state, eps=0.):
                if random.random() > eps:
                    state = torch.from_numpy(state).float() \
                        .unsqueeze(0).to(device)
                    self.model.eval()
                    with torch.no_grad():
                        action_values = self.model(state)
                    self.model.train()
                    return np.argmax(action_values.cpu().data.numpy())
                else:
                    return random.choice(np.arange(self.n_action))

        agent = Agent(n_state, n_action)

        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        epochs = 5000
        max_t = 5000
        eps_start = 1.0
        eps_end = 0.001
        eps_decay = 0.9995
        eps = eps_start

        for epoch in range(1, epochs + 1):
            state = env.reset()[0]
            state_size = env.observation_space.shape[0]

            score = 0
            for i in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                reward = reward if not done or score == 499 else -10
                agent.step(state, action, next_state, reward, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpoch {:4}\t Reward {:8.2f}\t Average Score: {:8.2f}'.format(epoch, score, \
                                                                                  np.mean(scores_window)), end="")
            if epoch % 100 == 0:
                print('\rEpoch {:4}\t Average Score: {:8.2f} \
                   \tEpsilon: {:8.3f}'.format(epoch, \
                                              np.mean(scores_window), eps))
            if epoch > 10 and np.mean(scores[-10:]) > 450:
                break

        plt.plot(scores)
        plt.title('Scores over increasing episodes')

        def play_game():
            done = False
            state = env.reset()[0]
            epoch = 0
            while (not done):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                env.render()
                state = next_state
                epoch += 1
            print(f"done, epoch:{epoch}")
            # env.close()

        play_game()

        return {}, {}

    @classmethod
    def M1_0_1(self, functionInfo):
        import copy
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
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
        pytorchModelFilePath = "Example/P36PyTorch/file/result/V1_0_1/9999/M1_0_1/Model.pth"

        # 儲存模型參數
        torch.save(model.state_dict(), pytorchModelFilePath)
        # Accuracy: 65.6%, Avg loss: 1.069365
        return {}, {"PyTorchModelFilePath":pytorchModelFilePath}

