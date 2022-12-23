import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon_local as executeOPSCommon

if __name__ == "__main__":
    basicInfo = {
        "RunType": ["runops"]
        , "Product": ["Example"]
        , "Project": ["P34PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = [9999]
    opsInfo["OPSOrderJson"] =  {
        "ExeFunctionArr": [
            "P0_0_1",
            "P0_0_2","M0_0_2",
            "P0_0_3","M0_0_3",
            "R0_0_4","P0_0_4","M0_0_4","UP0_0_4",
            "R0_0_5","P0_0_5","M0_0_5","UP0_0_5",
            "R0_0_6","P0_0_6","M0_0_6","UP0_0_6",
            "M0_0_7",
            "M0_0_8",
            "M0_0_9",
            "M0_0_10Train","M0_0_10Test",
            "M0_0_11Train","M0_0_11Test",
            "M0_0_12",
            "M0_0_13",
            "M0_0_14",
            "M0_0_15",
            "M0_0_16",
            "M0_0_17",
            "M0_0_18",
        ],
        "RepOPSRecordId": 9999,
        "RepFunctionArr": ["R0_0_6","P0_0_6",],
        "RunFunctionArr": ["M0_0_6","UP0_0_6",],
        "OrdFunctionArr": [
            {"Parent": "P0_0_2", "Child": "M0_0_2"},
            {"Parent": "P0_0_3", "Child": "M0_0_3"},
            {"Parent": "R0_0_4", "Child": "P0_0_4"},{"Parent": "P0_0_4", "Child": "M0_0_4"},{"Parent": "M0_0_4", "Child": "UP0_0_4"},
            {"Parent": "R0_0_5", "Child": "P0_0_5"},{"Parent": "P0_0_5", "Child": "M0_0_5"},{"Parent": "M0_0_5", "Child": "UP0_0_5"},
            {"Parent": "R0_0_6", "Child": "P0_0_6"},{"Parent": "P0_0_6", "Child": "M0_0_6"},{"Parent": "M0_0_6", "Child": "UP0_0_6"},
            {"Parent": "M0_0_10Train", "Child": "M0_0_10Test"},
            {"Parent": "M0_0_11Train", "Child": "M0_0_11Test"},
        ],
        "FunctionMemo": {
            "M0_0_1":"熟悉Pytorch的張量操作",
            "M0_0_2":"了解解決多元回歸問題",
            "M0_0_3":"使用單層NN解回歸問題",
            "M0_0_4":"使用NN預測歌曲發行年份",
            "M0_0_5":"使用NN預測信用卡範例",
            "M0_0_6":"使用NN預測花的多個分類",
            "M0_0_7":"使用CNN做手寫數字辨識，卷積、池化、全連接、Dropout",
            "M0_0_8":"使用訓練好的ResNet18進行圖片分類",
            "M0_0_9":"使用訓練好的ResNet18進行圖片分類",
            "M0_0_10Train":"使用CNN進行圖片分類",
            "M0_0_10Test":"使用訓練好的CNN進行圖片分類",
            "M0_0_11Train":"使用Resnet18進行圖片分類",
            "M0_0_11Test":"使用訓練好的Resnet18進行圖片分類",
            "M0_0_12":"使用RNN做時序預測",
            "M0_0_13":"使用LSTM作情感分析",
            "M0_0_14":"使用QLearning做路徑遊戲一",
            "M0_0_15":"使用QLearning做路徑遊戲二",
            "M0_0_16":"使用QLearning做GYM路徑遊戲",
            "M0_0_17":"使用QLearning做GYM滑車遊戲",
            "M0_0_18":"使用DQN做GYM車桿遊戲 ",
        },
    }
    opsInfo["ParameterJson"] = {
        "P0_0_1":{},
        "P0_0_2":{},"M0_0_2":{},
        "P0_0_3":{},"M0_0_3":{},
        "R0_0_4":{},"P0_0_4":{},"M0_0_4":{},"UP0_0_4":{},
        "R0_0_5":{},"P0_0_5":{},"M0_0_5":{},"UP0_0_5":{},
        "R0_0_6":{},"P0_0_6":{},"M0_0_6":{},"UP0_0_6":{},
        "M0_0_7":{},
        "M0_0_8":{},
        "M0_0_9":{},
        "M0_0_10Train":{}, "M0_0_10Test":{},
        "M0_0_11Train":{}, "M0_0_11Test":{},
        "M0_0_12":{},
        "M0_0_13":{},
        "M0_0_14":{},
        "M0_0_15":{},
        "M0_0_16":{},
        "M0_0_17":{},
        "M0_0_18":{},
    }
    executeOPSCommon.main(opsInfo)
