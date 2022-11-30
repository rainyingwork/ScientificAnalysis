import os

class StandardFunction():

    def __init__(self):
        pass

    # ==================================================              ==================================================

    @classmethod
    def insertOverwriteStandardData(self,product,project,tablename,dt,standardDataDF) :
        from dotenv import load_dotenv
        from package.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database="scientificanalysis"
            , schema="public"
        )
        deleteSQL = """
             DELETE FROM observationdata.standarddata
             WHERE 1 = 1 
                 AND product = '[:Product]'
                 AND project = '[:Project]'
                 AND tablename = '[:TableName]'
                 AND dt = '[:DT]'
        """.replace("[:Product]", product) \
            .replace("[:Project]", project) \
            .replace("[:TableName]", tablename) \
            .replace("[:DT]", dt)
        postgresCtrl.executeSQL(deleteSQL)
        self.insertStandardData(product, project, tablename, dt, standardDataDF)

    @classmethod
    def insertStandardData(self, product, project, tablename, dt, standardDataDF):
        from dotenv import load_dotenv
        from package.common.database.PostgresCtrl import PostgresCtrl

        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database="scientificanalysis"
            , schema="public"
        )

        standardDataDF['product'] = product
        standardDataDF['project'] = project
        standardDataDF['tablename'] = tablename
        standardDataDF['dt'] = dt

        tableFullName = "observationdata.standarddata"

        for column in StandardFunction.getStandardColumnNameArr():
            if column not in standardDataDF.columns:
                standardDataDF[column] = None
        insertTableInfoDF = postgresCtrl.getTableInfoDF(tableFullName)
        postgresCtrl.insertDataList(tableFullName, insertTableInfoDF, standardDataDF)


    # ==================================================              ==================================================

    @classmethod
    def getStandardColumnNameArr(self):
        columnNameArr = [
            "product", "project", "tablename", "dt"
            , "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "string_001", "string_002", "string_003", "string_004", "string_005"
            , "string_006", "string_007", "string_008", "string_009", "string_010"
            , "integer_001", "integer_002", "integer_003", "integer_004", "integer_005"
            , "integer_006", "integer_007", "integer_008", "integer_009", "integer_010"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "time_001", "time_002"
            , "json_001", "json_002"]
        return columnNameArr

    @classmethod
    def getDataColumnNameArr(self):
        columnNameArr = [
            "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "string_001", "string_002", "string_003", "string_004", "string_005"
            , "string_006", "string_007", "string_008", "string_009", "string_010"
            , "integer_001", "integer_002", "integer_003", "integer_004", "integer_005"
            , "integer_006", "integer_007", "integer_008", "integer_009", "integer_010"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "time_001", "time_002"
            , "json_001", "json_002"]
        return columnNameArr

    @classmethod
    def getTitleColumnNameArr(self):
        titleColumns = [
            "product", "project", "tablename", "dt"
        ]
        return titleColumns

    @classmethod
    def getCommonColumnNameArr(self):
        commonColumns = [
            "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
        ]
        return commonColumns

    @classmethod
    def getStringColumnNameArr(self):
        stringColumns = [
            "string_001", "string_002", "string_003", "string_004", "string_005"
            , "string_006", "string_007", "string_008", "string_009", "string_010"
        ]
        return stringColumns

    @classmethod
    def getDoubleColumnNameArr(self):
        doubleColumns = [
            "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
        ]
        return doubleColumns

    @classmethod
    def getDataTimeColumnNameArr(self):
        timeColumns = [
            "time_001", "time_002"
        ]
        return timeColumns

    @classmethod
    def getJsonColumnNameArr(self):
        jsonColumns = ["json_001", "json_002"]
        return jsonColumns

    # ==================================================              ==================================================

    @classmethod
    def getStandardColumnDocInfo(self):
        # 本段隨然可以簡短，但此代碼未來會很長複製，請勿特別簡短
        columnInfoMap = {}
        columnInfoMap["product"] = {"description": "product"}
        columnInfoMap["project"] = {"description": "project"}
        columnInfoMap["tablename"] = {"description": "tablename"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "common_001"}
        columnInfoMap["common_002"] = {"description": "common_002"}
        columnInfoMap["common_003"] = {"description": "common_003"}
        columnInfoMap["common_004"] = {"description": "common_004"}
        columnInfoMap["common_005"] = {"description": "common_005"}
        columnInfoMap["common_006"] = {"description": "common_006"}
        columnInfoMap["common_007"] = {"description": "common_007"}
        columnInfoMap["common_008"] = {"description": "common_008"}
        columnInfoMap["common_009"] = {"description": "common_009"}
        columnInfoMap["common_010"] = {"description": "common_010"}
        columnInfoMap["string_001"] = {"description": "string_001"}
        columnInfoMap["string_002"] = {"description": "string_002"}
        columnInfoMap["string_003"] = {"description": "string_003"}
        columnInfoMap["string_004"] = {"description": "string_004"}
        columnInfoMap["string_005"] = {"description": "string_005"}
        columnInfoMap["string_006"] = {"description": "string_006"}
        columnInfoMap["string_007"] = {"description": "string_007"}
        columnInfoMap["string_008"] = {"description": "string_008"}
        columnInfoMap["string_009"] = {"description": "string_009"}
        columnInfoMap["string_010"] = {"description": "string_010"}
        columnInfoMap["integer_001"] = {"description": "integer_001"}
        columnInfoMap["integer_002"] = {"description": "integer_002"}
        columnInfoMap["integer_003"] = {"description": "integer_003"}
        columnInfoMap["integer_004"] = {"description": "integer_004"}
        columnInfoMap["integer_005"] = {"description": "integer_005"}
        columnInfoMap["integer_006"] = {"description": "integer_006"}
        columnInfoMap["integer_007"] = {"description": "integer_007"}
        columnInfoMap["integer_008"] = {"description": "integer_008"}
        columnInfoMap["integer_009"] = {"description": "integer_009"}
        columnInfoMap["integer_010"] = {"description": "integer_010"}
        columnInfoMap["double_001"] = {"description": "double_001"}
        columnInfoMap["double_002"] = {"description": "double_002"}
        columnInfoMap["double_003"] = {"description": "double_003"}
        columnInfoMap["double_004"] = {"description": "double_004"}
        columnInfoMap["double_005"] = {"description": "double_005"}
        columnInfoMap["double_006"] = {"description": "double_006"}
        columnInfoMap["double_007"] = {"description": "double_007"}
        columnInfoMap["double_008"] = {"description": "double_008"}
        columnInfoMap["double_009"] = {"description": "double_009"}
        columnInfoMap["double_010"] = {"description": "double_010"}
        columnInfoMap["time_001"] = {"description": "time_001"}
        columnInfoMap["time_002"] = {"description": "time_002"}
        columnInfoMap["json_001"] = {"description": "json_001"}
        columnInfoMap["json_002"] = {"description": "json_002"}
        return columnInfoMap