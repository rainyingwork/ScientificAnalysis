
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
