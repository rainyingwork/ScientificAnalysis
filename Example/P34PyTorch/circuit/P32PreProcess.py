
class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import torch
        import numpy as np
        import glob # 多圖撈取模組
        from PIL import Image

        # torch.Tensor 浮點數張量，可用於GPU計算
        # torch.tensor 一般張量，視輸入而定

        example01 = torch.tensor(1,dtype=torch.int16)
        print(example01.shape)                                              # torch.Size([])
        print(example01)                                                    # tensor(1,dtype=torch.int16)

        example02 = torch.Tensor([1, 2, 3, 4, 5])
        print(example02.shape)                                              # torch.Size([5])
        print(example02)                                                    # tensor([1., 2., 3., 4., 5.]) 沒有dtype代表就是float，也會看到後面有.
        print(example02[:3])                                                # tensor([1., 2., 3.]) 操作的方式其實與一般操作很像
        print(example02[:-1])                                               # tensor([1., 2., 3., 4.])

        example03 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        print(example03.shape)                                              # torch.Size([2, 3])
        print(example03)                                                    # tensor([[1., 2., 3.],[4., 5., 6.]])

        example04 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        print(example04)                                                    # tensor([[1., 2., 3.],[4., 5., 6.]])
        print(example04.tolist())                                           # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 轉成純陣列
        print(example04.numel())                                            # 6 張量中元素的總數

        print(torch.arange(1, 6, 2))                                        # tensor([1,3,5]) torch.arange(start,end,step) 整數切
        print(torch.linspace(1, 10, 3))                                     # tensor([1.0,5.5,10.0]) torch.linspace(start,end,step) 均勻切
        print(torch.randn(2, 3))                                            # tensor([[2.05,-0.02,-0.17],[0.23,-0.28,-0.33]]) 為浮點數的X維張量
        print(torch.randperm(5))                                            # tensor([4, 2, 1, 3, 0]) 為整數的X維張量
        print(torch.eye(2, 3))                                              # tensor([[1., 0., 0.],[0., 1., 0.]]) 對角線為1的X維張量

        example05 = torch.arange(0, 6)
        print(example05)                                                    # tensor([0, 1, 2, 3, 4, 5]) torch.arange(start,end) 整數切
        print(example05.view(2, 3))                                         # tensor([[0, 1, 2],[3, 4, 5]]) 切成2個3列陣列
        print(example05.view(-1, 2))                                        # tensor([[0, 1],[2, 3],[4, 5]]) 切成n個2列陣列，n由系統運算

        example06 = example05.view(2, 3)
        example06 = example06.unsqueeze(1)                                  # 在軸1擴展維度
        print(example06)                                                    # tensor([[[0, 1, 2]],[[3, 4, 5]]])
        print(example06.shape)                                              # torch.Size([2, 1, 3])

        example07 = torch.arange(0, 6)
        example07 = example07.view(1, 1, 1, 2, 3)
        print(example07.shape)                                              # torch.Size([1, 1, 1, 2, 3])

        example08 = example07.squeeze(0)                                    # 降低軸0維度
        print(example08.shape)                                              # torch.Size([1, 1, 2, 3])
        example09 = example08.squeeze()                                     # 刪除維度為1的軸
        print(example09.shape)                                              # torch.Size([2, 3])

        example10 = torch.arange(0, 12)
        print(example10)                                                    # tensor([0,1,2,3,4,5,6,7,8,9,10,11])
        print(example10.resize_(2,6))                                       # tensor([[0,1,2,3,4,5],[6,7,8,9,10,11]])
        print(example10.resize_(1,6))                                       # tensor([0,1,2,3,4,5])
        print(example10.resize_(3,6))                                       # tensor([[0,1,2,3,4,5],[6,7,8,9,10,11],[4322,4550,43228,4322,432,432]]) 多出來的張量會給數字

        # 讀取單一圖檔
        pandaNP = np.array(Image.open('common/common/file/data/imgs/panda/panda1.jpg'))
        pandaTensor = torch.from_numpy(pandaNP)
        print(pandaTensor.shape)                                            # torch.Size([426, 640, 3])

        # 讀取多個圖檔
        pandaList = glob.glob("common/common/file/data/imgs/panda/*.jpg")
        print(pandaList)
        pandaNPList = []
        for panda in pandaList:
             tempPanda = Image.open(panda).resize((224, 224))
             pandaNPList.append(np.array(tempPanda))

        # 轉成多維向量
        pandaNPList = np.array(pandaNPList)
        pandaTensor = torch.from_numpy(pandaNPList)
        print(pandaTensor.shape)                                            # torch.Size([4, 224, 224, 3])

        return {}, {}

    @classmethod
    def P0_0_2(self, functionInfo):
        import torch
        torch.manual_seed(0)  # 設定隨機種子

        w = torch.tensor([1, 3, 5]).float()                                                 # 等同於 torch.Tensor([1, 3, 5])
        x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)],dim=1)                      # torch.cat 連接張量(需確認一下相關方式) torch.ones 為全1的X維張量 , torch.randn 為浮點數的X維張量
        # torch.mm() 是正常的矩陣相乘，(a,b) * (b,c) = (a,c)
        # torch.mv() 是矩陣與向量相乘，類似torch.mm()是(a,b) * (b,1) = (a,1)
        # torch.mul() 是矩陣的點乘，即對應的位相乘，會要求shape一樣,返回也是一樣大小的矩陣
        # torch.dot() 類似torch.mul()，但是是向量的對應位相乘在求和，返回一個tensor值
        y = torch.mv(x, w) + torch.randn(100) * 0.3
        print(x.shape, y.shape)                                                             # torch.Size([100, 3]) torch.Size([100])

        return {}, {'x': x, 'y': y}

    @classmethod
    def P0_0_3(self, functionInfo):
        import torch

        torch.manual_seed(0)  # 設定隨機種子

        # ========== RX_X_X ========== 相關說明可以參考 M0_0_2

        w = torch.tensor([1, 3, 5]).float()
        x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], dim=1)
        y = torch.mv(x, w) + torch.randn(100) * 0.3
        y = y.unsqueeze(1)  # 在軸1擴展維度
        print(x.shape, y.shape)

        return {}, {'x': x, 'y': y}

    @classmethod
    def P0_0_4(self, functionInfo):
        import copy
        import torch
        from sklearn.model_selection import train_test_split
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_4"])
        functionVersionInfo["Version"] = "P0_0_4"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        mainDF = globalObject["R0_0_4"]["ResultDF"]

        # 設定隨機種子
        torch.manual_seed(0)

        # 數據所有資料欄位
        msdColumns = mainDF.columns
        # 數據中所有為數值型態的資料欄位
        msdNumColumns = mainDF._get_numeric_data().columns
        print(list(set(msdColumns) - set(msdNumColumns)))  # [] -> 代表所有都是數值欄位
        print(mainDF.isnull().sum().sum())  # 0 -> 代表資料非常乾淨沒有空值

        outlierColumnList = []  # 確認雜訊欄位有哪一些
        for columnNum in range(mainDF.shape[1]):
            # 平均 +- 三倍標準差 (過濾雜訊)
            maxValue = mainDF[mainDF.columns[columnNum]].mean() + (3 * mainDF[mainDF.columns[columnNum]].std())
            minValue = mainDF[mainDF.columns[columnNum]].mean() - (3 * mainDF[mainDF.columns[columnNum]].std())
            noiseCount = 0
            for value in mainDF[mainDF.columns[columnNum]]:
                if value > maxValue or value < minValue:
                    noiseCount += 1
            noisePer = noiseCount / mainDF.shape[0]
            if noisePer > 0.05:  # 雜訊比例大於5%的盡量不要使用
                outlierColumnList.append(columnNum)
        print(outlierColumnList)  # [] 列出雜訊欄位

        x = mainDF.iloc[:, 1:]  # 欄位 1 ~ 90
        y = mainDF.iloc[:, 0]  # 欄位 0
        x = (x - x.mean()) / x.std()  # 訓練數據標準化

        print(x.head())

        # 拆分數據成2個子集，x_new : x_test = 80:20
        # 再拆分數據集x_new成2個子集, x_train : x_dev = 75:25
        xTrainDev, xTest, yTrainDev, yTest = train_test_split(x, y, test_size=0.2, random_state=0)
        xTrain, xDev, yTrain, yDev = train_test_split(xTrainDev, yTrainDev, test_size=0.25, random_state=0)
        print(xTrain.shape, xDev.shape, xTest.shape)  # (30000, 90) (10000, 90) (10000, 90)

        xTrainTensor = torch.tensor(xTrain.values).float()
        yTrainTensor = torch.tensor(yTrain.values).float().unsqueeze(1)  # 使用unsqueeze增加一個維度
        xDevTensor = torch.tensor(xDev.values).float()
        yDevTensor = torch.tensor(yDev.values).float().unsqueeze(1)  # 使用unsqueeze增加一個維度
        xTestTensor = torch.tensor(xTest.values).float()
        yTestTensor = torch.tensor(yTest.values).float().unsqueeze(1)  # 使用unsqueeze增加一個維度

        return {}, {
            "xTrainTensor": xTrainTensor,
            "yTrainTensor": yTrainTensor,
            "xDevTensor": xDevTensor,
            "yDevTensor": yDevTensor,
            "xTestTensor": xTestTensor,
            "yTestTensor": yTestTensor,
        }


    @classmethod
    def P1_0_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P1_0_1"])
        functionVersionInfo["Version"] = "P1_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        # 為資料處理保留位置
        trainData = globalObject[functionVersionInfo["DataVersion"]]["TrainData"]
        testData = globalObject[functionVersionInfo["DataVersion"]]["TestData"]
        return {}, {"TrainData":trainData,"TestData":testData}

