from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
class RawDataInfo(RawDataFunction) :
    pass

    @classmethod
    def getInfo_R0_0_1(self, makeInfo=None):

        tableInfo = {
            "Project": "P81DataPerception"
            , "TableName": "R0_0_1"
            , "Memo": "P81DataPerception"
            , "DataType": "RawData"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getAnalysisColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "專案"}
        columnInfoMap["project"] = {"description": "計畫"}
        columnInfoMap["version"] = {"description": "版本"}
        columnInfoMap["dt"] = {"description": "日期"}
        columnInfoMap["common_001"] = {"description": "Id"}
        columnInfoMap["double_001"] = {"description": "IsPurchaseCH"}
        columnInfoMap["double_002"] = {"description": "IsPurchaseMM"}
        columnInfoMap["double_003"] = {"description": "Store7"}
        columnInfoMap["double_004"] = {"description": "WeekofPurchase"}
        columnInfoMap["double_005"] = {"description": "StoreID"}
        columnInfoMap["double_006"] = {"description": "SpecialCH"}
        columnInfoMap["double_007"] = {"description": "SpecialMM"}
        columnInfoMap["double_008"] = {"description": "STORE"}
        columnInfoMap["double_009"] = {"description": "PriceCH"}
        columnInfoMap["double_010"] = {"description": "PriceMM"}
        columnInfoMap["double_011"] = {"description": "DiscCH"}
        columnInfoMap["double_012"] = {"description": "DiscMM"}
        columnInfoMap["double_013"] = {"description": "LoyalCH"}
        columnInfoMap["double_014"] = {"description": "SalePriceMM"}
        columnInfoMap["double_015"] = {"description": "SalePriceMM"}
        columnInfoMap["double_016"] = {"description": "PriceDiff"}
        columnInfoMap["double_017"] = {"description": "PctDiscMM"}
        columnInfoMap["double_018"] = {"description": "PctDiscCH"}

        return columnInfoMap