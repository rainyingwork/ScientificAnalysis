
class RawData() :

    @classmethod
    def R0_0_1(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as CustomerID
                , AA.common_002 as Country
                , AA.string_001 as InvoiceNo
                , AA.string_002 as StockCode
                , AA.string_003 as Description
                , AA.integer_001 as Quantity
                , AA.double_001 as UnitPrice
                , AA.time_001 as InvoiceDate
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RecommendSys'
                and AA.tablename = 'S0_0_1'
                and AA.dt = '20220101' 
                and AA.common_002 = 'France' ; 
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo, functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

