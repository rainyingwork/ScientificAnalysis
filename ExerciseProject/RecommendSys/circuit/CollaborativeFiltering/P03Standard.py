import pandas

class Standard () :

    @classmethod
    def S0_2_1(self, functionInfo):
        import pandas
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        dataFilePath = "ExerciseProject/RecommendSys/file/data/ml-100k/u.item"
        insertDataDF = pandas.read_csv(dataFilePath, sep='|', encoding='ISO-8859-1', names=['movie_id', 'movie_title'],usecols=[0, 1, ])
        insertDataDF.columns = ['common_001', 'string_001']
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RecommendSys", "S0_2_1", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}

    @classmethod
    def S0_2_2(self, functionInfo):
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        dataFilePath = "ExerciseProject/RecommendSys/file/data/ml-100k/u.data"
        insertDataDF = pandas.read_csv(dataFilePath, sep='\t', encoding='ISO-8859-1',names=["movie_id", "user_id",  "rating"], usecols=[0, 1, 2])
        insertDataDF.columns = ['common_001', 'common_002', 'integer_001']
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RecommendSys", "S0_2_2", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}
