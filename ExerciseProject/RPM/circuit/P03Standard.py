import pandas

class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        insertDataDF = pandas.read_csv("ExerciseProject/RPM/file/data/data.csv",converters={'CustomerID':str})
        insertDataDF['InvoiceNo'] = insertDataDF['InvoiceNo'].astype(str).str.replace(".0","")
        insertDataDF.columns = ['common_001', 'string_001', 'time_001', 'double_001']
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RPM", "S0_0_1", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}