

class StandardInfo ():

    @classmethod
    def getInfo_S0_0_1(self, makeInfo=None):
        from package.dataengineer.standard.StandardFunction import StandardFunction
        tableInfo = {
            "Project": "推薦系統"
            , "TableName": "S0_0_1"
            , "Memo": "推薦系統\n購買發票清單"
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

    @classmethod
    def getInfo_S0_1_1(self, makeInfo=None):
        from package.dataengineer.standard.StandardFunction import StandardFunction
        tableInfo = {
            "Project": "推薦系統"
            , "TableName": "S0_1_1"
            , "Memo": "推薦系統\n電影主要資料"
            , "DataType": "標準區"
            , "StartDate": "2022-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "推薦系統相關資料"
        }

        columnInfoMap = StandardFunction.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "ExerciseProject"}
        columnInfoMap["project"] = {"description": "RecommendSys"}
        columnInfoMap["tablename"] = {"description": "S0_1_1"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "電影編號", "memo": "movie_id"}
        columnInfoMap["string_001"] = {"description": "電影名稱", "memo": "title"}
        columnInfoMap["string_002"] = {"description": "電影演員", "memo": "cast"}
        columnInfoMap["string_003"] = {"description": "工作人員", "memo": "crew"}

        return columnInfoMap

    @classmethod
    def getInfo_S0_1_2(self, makeInfo=None):
        from package.dataengineer.standard.StandardFunction import StandardFunction
        tableInfo = {
            "Project": "推薦系統"
            , "TableName": "S0_1_2"
            , "Memo": "推薦系統\n電影細項資料"
            , "DataType": "標準區"
            , "StartDate": "2022-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "推薦系統相關資料"
        }

        columnInfoMap = StandardFunction.getStandardColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "ExerciseProject"}
        columnInfoMap["project"] = {"description": "RecommendSys"}
        columnInfoMap["tablename"] = {"description": "S0_1_2"}
        columnInfoMap["dt"] = {"description": "dt"}

        columnInfoMap["common_001"] = {"description": "電影編號","memo":"id"}
        columnInfoMap["common_006"] = {"description": "派別列表","memo":"genres"}
        columnInfoMap["common_007"] = {"description": "關鍵字列表","memo":"keywords"}
        columnInfoMap["common_008"] = {"description": "公司列表","memo":"production_companies"}
        columnInfoMap["common_009"] = {"description": "國家列表","memo":"production_countries"}
        columnInfoMap["common_010"] = {"description": "語言列表","memo":"spoken_languages"}
        columnInfoMap["string_001"] = {"description": "電影名稱","memo":"title"}
        columnInfoMap["string_002"] = {"description": "電影原始名稱","memo":"original_title"}
        columnInfoMap["string_003"] = {"description": "電影原始語言","memo":"original_language"}
        columnInfoMap["string_004"] = {"description": "電影標題","memo":"tagline"}
        columnInfoMap["string_005"] = {"description": "電影官網","memo":"homepage"}
        columnInfoMap["string_005"] = {"description": "電影說明","memo":"overview"}
        columnInfoMap["string_010"] = {"description": "狀態","memo":"status"}
        columnInfoMap["integer_001"] = {"description": "預算","memo":"budget"}
        columnInfoMap["integer_002"] = {"description": "收入","memo":"revenue"}
        columnInfoMap["double_001"] = {"description": "評級平均星數","memo":"vote_average"}
        columnInfoMap["double_002"] = {"description": "評級數量","memo":"vote_count"}
        columnInfoMap["double_003"] = {"description": "電影知名度","memo":"popularity"}
        columnInfoMap["double_004"] = {"description": "電影時長","memo":"runtime"}
        columnInfoMap["time_001"] = {"description": "電影發行日","memo":"release_date"}

        return columnInfoMap