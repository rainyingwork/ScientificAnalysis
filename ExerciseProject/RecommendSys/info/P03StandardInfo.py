

class StandardInfo ():

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):
        from package.dataengineer.standard.StandardFunction import StandardFunction
        tableInfo = {
            "Project": "推薦系統"
            , "TableName": "S0_0_1"
            , "Memo": "推薦系統\n關聯分析相關資料"
            , "DataType": "標準區"
            , "StartDate": "2022-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "推薦系統相關資料"
        }

        columnInfoMap = StandardFunction.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "ExerciseProject"}
        columnInfoMap["project"] = {"description": "RecommendSys"}
        columnInfoMap["tablename"] = {"description": "S0_0_1"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "客戶ID","memo":"CustomerID"}
        columnInfoMap["common_002"] = {"description": "國家","memo":"Country"}
        columnInfoMap["string_001"] = {"description": "發票編號","memo":"InvoiceNo"}
        columnInfoMap["string_002"] = {"description": "產品編號","memo":"StockCode"}
        columnInfoMap["string_003"] = {"description": "產品說明","memo":"Description"}
        columnInfoMap["integer_001"] = {"description": "產品數量","memo":"Quantity"}
        columnInfoMap["double_001"] = {"description": "產品單價","memo":"UnitPrice"}
        columnInfoMap["time_001"] = {"description": "發票日期","memo":"InvoiceDate"}

        return columnInfoMap