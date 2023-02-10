import os
from package.common.common.database.PostgresCtrl import PostgresCtrl
from dotenv import load_dotenv
import pandas

load_dotenv(dotenv_path="env/postgresql.env")

class EntityBase (object):

    def __init__(self):
        self.postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        self.entity = {}
        self.schemaName = ""
        self.tableName = ""
        self.tableInfoDF = None

    def getEntityId(self):
        return self.entity["{}id".format(self.tableName)]

    def getEntity(self):
        return self.entity

    def setEntity(self, entity):
        self.entity = {}
        for tableInfoIndex, tableInfoRow in self.tableInfoDF.iterrows():
            if tableInfoRow["columnname"] in entity.keys() :
                self.entity[tableInfoRow["columnname"]] = entity[tableInfoRow["columnname"]]

    def getColumnValue(self , columnName):
        return self.entity[columnName]

    def setColumnValue(self, columnName , columnValue):
        self.entity[columnName] = columnValue

    def insertEntity(self):
        if "{}id".format(self.tableName) not in self.entity.keys():
            self.entity["{}id".format(self.tableName)] = self.getNextPrimaryKeyId()
        self.postgresCtrl.insertData(tableFullName="{}.{}".format(self.schemaName,self.tableName), insertTableInfoDF=self.tableInfoDF , insertData=self.entity)

    def updateEntity(self):
        self.postgresCtrl.updateData(tableFullName="{}.{}".format(self.schemaName, self.tableName), updateTableInfoDF=self.tableInfoDF, updateData=self.entity)

    def insertEntityList(self , entityList):
        newEntityList = []
        for entity in entityList :
            entity["{}id".format(self.tableName)] = self.getNextPrimaryKeyId()
            newEntityList.append(entity)
        df = pandas.DataFrame(newEntityList)
        self.postgresCtrl.insertData(tableFullName="{}.{}".format(self.schemaName, self.tableName), insertTableInfoDF=self.tableInfoDF, insertData=df)

    def getNextPrimaryKeyId(self):
        sql = " SELECT nextval('{}.{}_{}id_seq') ".format(self.schemaName, self.tableName, self.tableName)
        df = self.postgresCtrl.searchSQL(sql)
        return df['nextval'][0]

    def getEntityByPrimaryKeyId(self, primaryKeyId):
        sql = """
           SELECT * 
           FROM [:schemaName].[:TableName] AA
           WHERE 1 = 1 
               AND AA.[:TableName]id = [:PrimaryKeyId]
           limit 1 
        """.replace('[:schemaName]', self.schemaName) \
            .replace('[:TableName]', self.tableName) \
            .replace('[:PrimaryKeyId]', str(primaryKeyId))
        df = self.postgresCtrl.searchSQL(sql)
        self.setEntity(df.iloc[0].to_dict())
        return self.entity

