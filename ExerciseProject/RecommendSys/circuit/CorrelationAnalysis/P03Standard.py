import pandas

class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        insertDataDF = pandas.read_excel('ExerciseProject/RecommendSys/file/data/Online Retail.xlsx', converters = {'CustomerID': str})
        insertDataDF['InvoiceNo'] = insertDataDF['InvoiceNo'].astype(str).str.replace(".0", "")
        insertDataDF.columns = [
            'string_001' , 'string_002' , 'string_003'
            , 'integer_001'
            , 'time_001'
            , 'double_001'
            , 'common_001'
            , 'common_002'
        ]
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RecommendSys", "S0_0_1", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}
