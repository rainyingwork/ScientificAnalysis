import pandas

class Standard () :

    @classmethod
    def S0_1_1(self, functionInfo):
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        insertDataDF = pandas.read_csv('ExerciseProject/RecommendSys/file/data/tmdb_5000_credits.csv', converters = {'movie_id': str})
        insertDataDF.columns = ['common_001' , 'string_001' , 'common_009' , 'common_010']
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RecommendSys", "S0_1_1", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}

    @classmethod
    def S0_1_2(self, functionInfo):
        from package.dataengineer.common.standard.StandardFunction import StandardFunction
        insertDataDF = pandas.read_csv('ExerciseProject/RecommendSys/file/data/tmdb_5000_movies.csv', converters = {'id': str})
        insertDataDF['vote_count'] = insertDataDF['vote_count'].astype(float)
        insertDataDF.columns = [
            'integer_001', 'common_006', 'string_005', 'common_001', 'common_007'
            , 'string_003', 'string_002', 'string_006', 'double_003', 'common_008'
            , 'common_009', 'time_001', 'integer_002', 'double_004', 'common_010'
            , 'string_010', 'string_004', 'string_001', 'double_001', 'double_002'
        ]
        StandardFunction.insertOverwriteStandardData("ExerciseProject", "RecommendSys", "S0_1_2", "20220101", insertDataDF,useType="IO")
        return {"result": "OK"}, {}
