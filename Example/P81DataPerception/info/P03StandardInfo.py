from package.dataengineer.common.standard.StandardFunction import StandardFunction

class StandardInfo(StandardFunction):

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):
        tableInfo = {
            "Project": "P81DataPerception"
            , "TableName": "S0_0_1"
            , "Memo": "P81DataPerception"
            , "DataType": "Standard"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getInfo_S0_0_X(makeInfo)
        columnInfoMap["TableInfo"] = tableInfo

        return columnInfoMap

    @classmethod
    def getInfo_S0_0_11(self, makeInfo=None):
        tableInfo = {
            "Project": "P81DataPerception"
            , "TableName": "S0_0_11"
            , "Memo": "P81DataPerception"
            , "DataType": "Standard"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getInfo_S0_0_X(makeInfo)
        columnInfoMap["TableInfo"] = tableInfo

        return columnInfoMap

    @classmethod
    def getInfo_S0_0_12(self, makeInfo=None):
        tableInfo = {
            "Project": "P81DataPerception"
            , "TableName": "S0_0_12"
            , "Memo": "P81DataPerception"
            , "DataType": "Standard"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getInfo_S0_0_X(makeInfo)
        columnInfoMap["TableInfo"] = tableInfo

        return columnInfoMap

    @classmethod
    def getInfo_S0_0_X(self, makeInfo=None):

        checkFuncStrs = ["IsNotNull", "IsNotNullPer", "Count", "DisCount"]
        checkFuncInts = ["IsNotNull", "IsNotNullPer", "Count", "DisCount",
                         "Avg", "Round", "Max", "Min", "Sum"]

        columnInfoMap = self.getStandardColumnDocInfo()
        columnInfoMap["product"] = {"description": "product", 'datatype': 'string'}
        columnInfoMap["project"] = {"description": "project", 'datatype': 'string'}
        columnInfoMap["tablename"] = {"description": "tablename", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "dt", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "Id", 'datatype': 'string', "checkfuncs": checkFuncStrs}
        columnInfoMap["string_001"] = {"description": "Purchase", 'datatype': 'string', "checkfuncs": checkFuncStrs}
        columnInfoMap["string_002"] = {"description": "Store7", 'datatype': 'string', "checkfuncs": checkFuncStrs}
        columnInfoMap["integer_001"] = {"description": "WeekofPurchase", 'datatype': 'integer',"checkfuncs": checkFuncInts}
        columnInfoMap["integer_002"] = {"description": "StoreID", 'datatype': 'integer', "checkfuncs": checkFuncInts}
        columnInfoMap["integer_003"] = {"description": "SpecialCH", 'datatype': 'integer', "checkfuncs": checkFuncInts}
        columnInfoMap["integer_004"] = {"description": "SpecialMM", 'datatype': 'integer', "checkfuncs": checkFuncInts}
        columnInfoMap["integer_005"] = {"description": "STORE", 'datatype': 'integer', "checkfuncs": checkFuncInts}
        columnInfoMap["double_001"] = {"description": "PriceCH", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_002"] = {"description": "PriceMM", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_003"] = {"description": "DiscCH", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_004"] = {"description": "DiscMM", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_005"] = {"description": "LoyalCH", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_006"] = {"description": "SalePriceMM", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_007"] = {"description": "SalePriceMM", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_008"] = {"description": "PriceDiff", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_009"] = {"description": "PctDiscMM", 'datatype': 'double', "checkfuncs": checkFuncInts}
        columnInfoMap["double_010"] = {"description": "PctDiscCH", 'datatype': 'double', "checkfuncs": checkFuncInts}

        return columnInfoMap