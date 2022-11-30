import copy

class RawData() :

    @classmethod
    def R0_0_1(self, functionInfo):
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
        functionVersionInfo["SQLStrs"] = """
            select
                AA.common_001 as customerid
                , AA.string_001 as invoiceno
                , AA.time_001 as invoicedate
                , AA.double_001 as totalprice
            from observationdata.standarddata AA
            where 1 = 1
                and AA.product = 'ExerciseProject'
                and AA.project = 'RPM'
                and AA.tablename = 'S0_0_1'
                and AA.dt = '20220101' ; 
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo, functionVersionInfo)
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

