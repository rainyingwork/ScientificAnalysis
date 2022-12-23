
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
