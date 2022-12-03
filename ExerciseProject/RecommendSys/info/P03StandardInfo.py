

class StandardInfo ():

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):
        from package.dataengineer.standard.StandardFunction import StandardFunction
        tableInfo = {
            "Project": "RecommendSys"
            , "TableName": "S0_0_1"
            , "Memo": "RecommendSys_CorrelationAnalysis"
            , "DataType": "Standard"
            , "StartDate": "2022-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "RecommendSys_CorrelationAnalysis"
        }

        columnInfoMap = StandardFunction.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "product"}
        columnInfoMap["project"] = {"description": "project"}
        columnInfoMap["tablename"] = {"description": "tablename"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "CustomerID"}
        columnInfoMap["common_002"] = {"description": "Country"}
        columnInfoMap["string_001"] = {"description": "InvoiceNo"}
        columnInfoMap["string_002"] = {"description": "StockCode"}
        columnInfoMap["integer_001"] = {"description": "Quantity"}
        columnInfoMap["double_001"] = {"description": "UnitPrice"}
        columnInfoMap["time_001"] = {"description": "InvoiceDate"}

        return columnInfoMap