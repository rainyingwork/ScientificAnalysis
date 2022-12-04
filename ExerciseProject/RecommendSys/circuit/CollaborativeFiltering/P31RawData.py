
class RawData() :

    @classmethod
    def R0_2_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_2_1"])
        functionVersionInfo["Version"] = "R0_2_1"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as MovieID
                , AA.string_001 as MovieTitle
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RecommendSys'
                and AA.tablename = 'S0_2_1'
                and AA.dt = '20220101' ;
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo, functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_2_2(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_2_2"])
        functionVersionInfo["Version"] = "R0_2_2"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as MovieID
                , AA.common_002 as UserID
                , AA.Integer_001 as Rating
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RecommendSys'
                and AA.tablename = 'S0_2_2'
                and AA.dt = '20220101' ;
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo,functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict




