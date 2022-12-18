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
            'M0_0_1','M0_0_2','M0_0_3','M0_0_4','M0_0_5',
            'M0_0_6','M0_0_7','M0_0_8','M0_0_9',
            'M0_0_10Train','M0_0_10Test',
            'M0_0_11Train','M0_0_11Test',
            'M0_0_12','M0_0_13','M0_0_14','M0_0_15',
            'M0_0_16','M0_0_17','M0_0_18']
        , "RepOPSRecordId": 9999
        # , "RepFunctionArr": ["R1_0_1","P1_0_1","M1_0_1"]
        , "RunFunctionArr": ["M0_0_9"]
        , "OrdFunctionArr": [
            {"Parent": "M0_0_10Train", "Child": "M0_0_10Test"}
            , {"Parent": "M0_0_11Train", "Child": "M0_0_11Test"}
        ]
        , "FunctionMemo": {
            "M0_0_1":"熟悉Pytorch的張量操作"
            , "M0_0_2":"了解解決多元回歸問題"
            , "M0_0_3":"單層神經網路解回歸問題"
            , "M0_0_4":"預測歌曲發行年份"
            , "M0_0_5":"信用卡範例"
            , "M0_0_6":"花的多元分類"
            , "M0_0_7":"手寫數字辨識"
            , "M0_0_8":"訓練好的圖片分類"

            , "M0_0_9":"熟悉Pytorch的張量操作"
            , "M0_0_10Train":"熟悉Pytorch的張量操作"
            , "M0_0_10Test":"熟悉Pytorch的張量操作"
            , "M0_0_11Train":"熟悉Pytorch的張量操作"
            , "M0_0_11Test":"熟悉Pytorch的張量操作"
            , "M0_0_12":"熟悉Pytorch的張量操作"
            , "M0_0_13":"熟悉Pytorch的張量操作"
            , "M0_0_14":"熟悉Pytorch的張量操作"
            , "M0_0_15":"熟悉Pytorch的張量操作"
            , "M0_0_16":"熟悉Pytorch的張量操作"
            , "M0_0_17":"熟悉Pytorch的張量操作"
            , "M0_0_18":"熟悉Pytorch的張量操作"
        }
    }
    opsInfo["ParameterJson"] = {}
    executeOPSCommon.main(opsInfo)

"""

    

      


"""