import sqlalchemy
import psycopg2
import pandas
import polars
import io

class PostgresCtrl:

    # 初始化
    def __init__(self, host=None, port=None, user=None, password=None, database=None, schema=None):
        self.__host = host
        self.__port = port
        self.__user = user
        self.__password = password
        self.__database = database
        self.__schema = schema
        self.__connstr = "postgresql://{}:{}@{}:{}/{}".format(self.__user, self.__password, self.__host, self.__port, self.__database)

    # 執行SQL
    def executeSQL(self, sql):
        connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password, host=self.__host, port=self.__port)
        cursor = connect.cursor()
        cursor.execute(sql)
        cursor.execute("commit;")
        connect.close()

    # 查詢SQL，回傳Dataframe格式的資料
    def searchSQL(self, sql , readDerviedClass = 'pandas' , writeDerviedClass = 'pandas'):
        if readDerviedClass == 'pandas':
            connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password, host=self.__host, port=self.__port)
            searchDataDF = pandas.read_sql(sql, connect)
            connect.close()
            return searchDataDF
        elif readDerviedClass == 'polars':
            connect = self.__connstr
            searchDataDF = polars.read_database(sql, connect)
            if writeDerviedClass == 'pandas':
                searchDataDF = searchDataDF.to_pandas()
            else :
                pass
            return searchDataDF

    # 將Entity，使用語法方式，直接塞入資料庫指定Table
    def insertData(self, tableFullName, insertTableInfoDF, insertData):
        insertDataDF = pandas.DataFrame([insertData])
        self.insertDataList(tableFullName, insertTableInfoDF, insertDataDF)

    # 將Dataframe，使用語法方式，直接塞入資料庫指定Table
    def insertDataList(self, tableFullName, insertTableInfoDF, insertDataDF, insertMaxCount=200):
        connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password, host=self.__host, port=self.__port)
        cursor = connect.cursor()
        schemaName = tableFullName.split(".")[0] if tableFullName.find(".") >= 0 else self.__schema
        tableName = tableFullName.split(".")[1] if tableFullName.find(".") >= 0 else tableFullName

        insertCodeInitStr = "INSERT INTO [:TableName] ( [:Columns] ) VALUES [:Values] "
        columnsStr = ""
        valuesListStr = ""
        for infoIndex, infoRow in insertTableInfoDF.iterrows():
            if columnsStr == "":
                columnsStr = infoRow["columnname"]
            else:
                columnsStr = columnsStr + "," + infoRow["columnname"]

        insertCount = 0
        for dataIndex, dataRow in insertDataDF.iterrows():
            valuesStr = ""
            for infoIndex, infoRow in insertTableInfoDF.iterrows():
                columnName = infoRow['columnname']
                insertType = infoRow['inserttype']
                valueStr = ""
                if valuesStr != "":
                    valueStr = ","
                if (insertType == "String" or insertType == "Date" or insertType == "Time") and dataRow[columnName] != None:
                    valueStr = valueStr + "'{}'"
                else:
                    valueStr = valueStr + "{}"
                if dataRow[columnName] == None:
                    dataValueStr = "null"
                else:
                    dataValueStr = str(dataRow[columnName])
                valuesStr = valuesStr + valueStr.format(dataValueStr.replace("'", "''"))
            if valuesListStr == "":
                valuesStr = "(" + valuesStr + ")"
            else:
                valuesStr = ",(" + valuesStr + ")"
            valuesListStr = valuesListStr + "\n" + valuesStr
            insertCount = insertCount + 1
            if insertCount >= insertMaxCount:
                insertCodeStr = insertCodeInitStr.replace("[:TableName]", "{}.{}".format(schemaName, tableName))
                insertCodeStr = insertCodeStr.replace("[:Columns]", columnsStr)
                insertCodeStr = insertCodeStr.replace("[:Values]", valuesListStr)
                cursor.execute(insertCodeStr)
                # print("已寫入 {} {} 筆資料".format(tableName, str(insertCount)))
                valuesListStr = ""
                insertCount = 0

        if insertCount > 0:
            insertCodeStr = insertCodeInitStr.replace("[:TableName]", "{}.{}".format(schemaName, tableName))
            insertCodeStr = insertCodeStr.replace("[:Columns]", columnsStr)
            insertCodeStr = insertCodeStr.replace("[:Values]", valuesListStr)
            cursor.execute(insertCodeStr)
            # print("已寫入 {} {} 筆資料".format(tableName, str(insertCount)))

        cursor.execute("commit;")
        connect.close()

    # 將Dataframe，使用IO方式，直接塞入資料庫指定Table
    def insertDataByIO(self, tableFullName, insertTableInfoDF, insertDataDF, ifExists="fail"):
        from urllib.parse import quote_plus as urlquote
        postgreEngine = sqlalchemy.create_engine("postgresql://" + self.__user + ":" + urlquote(self.__password) + "@" + self.__host + ":" + str(self.__port) + "/" + self.__database)
        stringDataIO = io.StringIO()
        insertDataDF.to_csv(stringDataIO, sep="\t", index=False)
        stringDataIO.seek(0)
        with postgreEngine.connect() as connection:
            with connection.connection.cursor() as cursor:
                copyCmd = "COPY %s FROM STDIN HEADER DELIMITER '\t' CSV" % tableFullName
                cursor.copy_expert(copyCmd, stringDataIO)
            connection.connection.commit()

    # 將Entity，使用語法方式，更新資料庫指定TableID
    def updateData(self, tableFullName, updateTableInfoDF, updateData):
        updateDataDF = pandas.DataFrame([updateData])
        self.updateDataList( tableFullName, updateTableInfoDF, updateDataDF)

    # 將Dataframe，使用語法方式，更新資料庫指定TableID
    def updateDataList(self, tableFullName, updateTableInfoDF, updateDataDF):
        connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password,host=self.__host, port=self.__port)
        cursor = connect.cursor()
        schemaName = tableFullName.split(".")[0] if tableFullName.find(".") >= 0 else self.__schema
        tableName = tableFullName.split(".")[1] if tableFullName.find(".") >= 0 else tableFullName
        updateCodeInitStr = "UPDATE [:TableName] SET [:ColumnValues] WHERE [:Where] "
        for dataIndex, dataRow in updateDataDF.iterrows():
            columnsValuesStr = ""
            for infoIndex, infoRow in updateTableInfoDF.iterrows():
                columnName = infoRow['columnname']
                insertType = infoRow['inserttype']
                if columnName in ["createtime", "deletetime", "{}id".format(tableName)]:
                    continue
                if columnsValuesStr != "":
                    columnsValuesStr = columnsValuesStr + ","
                if columnName == "modifytime":
                    columnsValuesStr = columnsValuesStr + "modifytime = now()"
                elif (insertType == "String" or insertType == "Date" or insertType == "Time") and dataRow[columnName] != None:
                    columnsValuesStr = columnsValuesStr + "{} = '{}'".format(infoRow["columnname"],dataRow[columnName])
                elif dataRow[columnName] != None:
                    columnsValuesStr = columnsValuesStr + "{} = {}".format(infoRow["columnname"],dataRow[columnName])
                else:
                    columnsValuesStr = columnsValuesStr + "{} = null".format(infoRow["columnname"])
            whereStr = "{}id = {}".format(tableName, dataRow["{}id".format(tableName)])
            updateCodeStr = updateCodeInitStr.replace("[:TableName]", "{}.{}".format(schemaName, tableName))
            updateCodeStr = updateCodeStr.replace("[:ColumnValues]", columnsValuesStr)
            updateCodeStr = updateCodeStr.replace("[:Where]", whereStr)
            cursor.execute(updateCodeStr)
            # print("已更新 {} 1 筆資料".format(tableName))
            cursor.execute("commit;")
        connect.close()

    # 確認資料表是否存在
    def isTableExist(self, tableName):
        connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password,host=self.__host, port=self.__port)
        cursor = connect.cursor()
        cursor.execute("SELECT * FROM information_schema.tables where table_schema = '" + self.__schema + "' AND table_name = '" + tableName + "'")
        is_exist = False
        if cursor.rowcount > 0:
            is_exist = True
        connect.close()
        return is_exist

    # 取得資料表資訊
    def getTableInfoDF(self, tableName):
        connect = psycopg2.connect(database=self.__database, user=self.__user, password=self.__password, host=self.__host,port=self.__port)
        schemaName = self.__schema
        if tableName.find(".") >= 0 :
            schemaName = tableName.split(".")[0]
            tableName = tableName.split(".")[1]

        sql = """
            SELECT 
                AA.table_schema as tableschema
                , AA.table_name as tablename 
                , AA.ordinal_position as position
                , AA.column_name as columnname
                , AA.data_type as datatype
                , CASE 
                    WHEN AA.data_type like '%timestamp%' THEN 'Time'
                    WHEN AA.data_type like '%int%' THEN 'Integer'
                    WHEN AA.data_type like '%double%' THEN 'Double'
                    ELSE 'String' 
                  END as inserttype
            FROM information_schema.columns AA
            WHERE 1 = 1 
                AND AA.table_schema = '[:SchemaName]'
                AND AA.table_name = '[:TableName]'
            ORDER BY 
                ordinal_position ;
        """.replace("[:SchemaName]",schemaName).replace("[:TableName]",tableName)
        df = pandas.read_sql(sql, connect)
        connect.close()
        return df





