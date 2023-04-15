from package.dataengineer.common.standard.StandardFunction import StandardFunction

class StandardInfo(StandardFunction):

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):

        tableInfo = {
            "Project": "P08DataCheck"
            , "TableName": "S0_0_1"
            , "Memo": "P08DataCheck"
            , "DataType": "Standard"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        checkFuncStrs = ["IsNotNull", "IsNotNullPer", "Count", "DisCount"]
        checkFuncInts = ["IsNotNull", "IsNotNullPer", "Count", "DisCount",
                         "Percentile", "Count", "DisCount", "Avg", "Round", "Max", "Min", "Sum"]

        columnInfoMap = self.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "product"}
        columnInfoMap["project"] = {"description": "project"}
        columnInfoMap["tablename"] = {"description": "tablename"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "Id", "checkfuncs": checkFuncStrs}
        columnInfoMap["string_001"] = {"description": "Purchase", "checkfuncs": checkFuncInts}
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
