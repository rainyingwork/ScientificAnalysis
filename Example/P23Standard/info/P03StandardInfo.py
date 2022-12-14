from package.dataengineer.common.standard.StandardFunction import StandardFunction

class StandardInfo(StandardFunction):

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):

        tableInfo = {
            "Project": "P13Standard"
            , "TableName": "S0_0_1"
            , "Memo": "P13Standard"
            , "DataType": "Standard"
            , "StartDate": "2020-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "product"}
        columnInfoMap["project"] = {"description": "project"}
        columnInfoMap["tablename"] = {"description": "tablename"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "Id"}
        columnInfoMap["string_001"] = {"description": "Purchase"}
        columnInfoMap["string_002"] = {"description": "Store7"}
        columnInfoMap["integer_001"] = {"description": "WeekofPurchase"}
        columnInfoMap["integer_002"] = {"description": "StoreID"}
        columnInfoMap["integer_003"] = {"description": "SpecialCH"}
        columnInfoMap["integer_004"] = {"description": "SpecialMM"}
        columnInfoMap["integer_005"] = {"description": "STORE"}
        columnInfoMap["double_001"] = {"description": "PriceCH"}
        columnInfoMap["double_002"] = {"description": "PriceMM"}
        columnInfoMap["double_003"] = {"description": "DiscCH"}
        columnInfoMap["double_004"] = {"description": "DiscMM"}
        columnInfoMap["double_005"] = {"description": "LoyalCH"}
        columnInfoMap["double_006"] = {"description": "SalePriceMM"}
        columnInfoMap["double_007"] = {"description": "SalePriceMM"}
        columnInfoMap["double_008"] = {"description": "PriceDiff"}
        columnInfoMap["double_009"] = {"description": "PctDiscMM"}
        columnInfoMap["double_010"] = {"description": "PctDiscCH"}

        return columnInfoMap
