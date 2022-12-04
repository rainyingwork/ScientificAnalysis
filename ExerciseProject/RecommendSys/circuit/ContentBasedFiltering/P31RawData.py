
class RawData() :

    @classmethod
    def R0_1_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_1_1"])
        functionVersionInfo["Version"] = "R0_1_1"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as MovieID
                , AA.string_001 as Title
                , AA.common_009 as Cast
                , AA.common_010 as Crew
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RecommendSys'
                and AA.tablename = 'S0_1_1'
                and AA.dt = '20220101' ; 
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo, functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_1_2(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_1_2"])
        functionVersionInfo["Version"] = "R0_1_2"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as MovieID
                , AA.common_006 as Genres
                , AA.common_007 as Keywords
                , AA.common_008 as ProductionCompanies
                , AA.common_009 as ProductionCountries
                , AA.common_010 as SpokenLanguages
                , AA.string_001 as Title
                , AA.string_002 as OriginalTitle
                , AA.string_003 as OriginalLanguage
                , AA.string_004 as Tagline
                , AA.string_005 as Homepage
                , AA.string_006 as Overview
                , AA.string_010 as Status
                , AA.integer_001 as Budget
                , AA.integer_002 as Revenue
                , AA.double_001 as VoteAverage
                , AA.double_002 as VoteCount
                , AA.double_003 as Popularity
                , AA.double_004 as Runtime
                , AA.time_001 as ReleaseDate
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RecommendSys'
                and AA.tablename = 'S0_1_2'
                and AA.dt = '20220101' ; 
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo,functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict




