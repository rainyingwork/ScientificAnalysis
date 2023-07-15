
class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        import copy
        import random
        from pycaret.datasets import get_data
        from sklearn.model_selection import train_test_split
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        oriInsertDataDF = get_data('juice')
        fvInfo = copy.deepcopy(functionInfo["ParameterJson"]["S0_0_1"])
        insertDataDF , _ = train_test_split(oriInsertDataDF, test_size=random.randrange(100,200)/1000)
        insertDataDF.columns = [
            'common_001'
            ,'string_001'
            ,'integer_001','integer_002'
            ,'double_001','double_002','double_003','double_004'
            ,'integer_003','integer_004'
            ,'double_005','double_006','double_007','double_008'
            ,'string_002'
            ,'double_009','double_010','double_011'
            ,'integer_005'
        ]
        StandardFunction.insertOverwriteStandardData("Example","P81DataPerception","S0_0_1",fvInfo["DataTime"].replace("-",""),insertDataDF)
        return {"result": "OK"} , {}



