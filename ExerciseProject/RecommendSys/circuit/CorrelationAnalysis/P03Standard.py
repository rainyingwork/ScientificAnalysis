import os
import pandas
from package.dataengineer.standard.StandardFunction import StandardFunction

class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
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

        # InvoiceNo string_001
        # StockCode string_002
        # Description string_003
        # Quantity integer_001
        # InvoiceDate time_001
        # UnitPrice  double_001
        # CustomerID common_001
        # Country common_002