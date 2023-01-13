
class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        from pycaret.datasets import get_data
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        insertDataDF = get_data('juice')
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
        StandardFunction.insertOverwriteStandardData("Example","P23Standard","S0_0_1","20220101",insertDataDF)
        return {"result": "OK"} , {}