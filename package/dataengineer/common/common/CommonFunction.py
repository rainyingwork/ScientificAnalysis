import os
import datetime
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl

class CommonFunction():

    def __init__(self):
        pass

    # ==================================================  ExeSQLStrs ==================================================

    @classmethod
    def makeExeSQLStrsByDataBase(self, fvInfo, otherInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        sqlStrs = fvInfo['SQLStrs']
        sqlReplaceArr = fvInfo["SQLReplaceArr"]
        for sqlReplace in sqlReplaceArr:
            sqlStrs = sqlStrs.replace(sqlReplace[0], sqlReplace[1])
        sqlStrArr = sqlStrs.split(";")[:-1]
        for sqlStr in sqlStrArr:
            postgresCtrl.executeSQL(sqlStr)
        return {}

    # ==================================================    Common    ==================================================

    @classmethod
    def getCommonSQLReplaceArr(self,functionInfo , functionVersionInfo) :
        sqlReplaceArr = [
            ["[:Product]", functionInfo['Product']]
            , ["[:Project]", functionInfo['Project']]
            , ["[:OPSVersion]", functionInfo['OPSVersion']]
            , ["[:FunctionVersion]", functionVersionInfo['Version']]
        ]
        if "DataTime" in functionVersionInfo.keys() :
            makeTime = datetime.datetime.strptime(functionVersionInfo["DataTime"], "%Y-%m-%d")
            sqlReplaceArr.append(["[:DateLine]", makeTime.strftime("%Y-%m-%d")])
            sqlReplaceArr.append(["[:DateLine]", makeTime.strftime("%Y-%m-%d")])
        return sqlReplaceArr

