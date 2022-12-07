import os
import datetime
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.common.AnalysisFunction import AnalysisFunction

class CommonFunction(AnalysisFunction):

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
            , database="scientificanalysis"
            , schema="public"
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

    @classmethod
    def getDatabaseResultJson(self,functionInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database="scientificanalysis"
            , schema="public"
        )
        databaseProduct = functionInfo['DatabaseProduct']
        databaseProject = functionInfo['DatabaseProject']
        databaseOPSVersion = functionInfo['DatabaseOPSVersion']
        databaseOPSRecord = functionInfo['DatabaseOPSRecord']
        databaseFunction = functionInfo['DatabaseFunction']
        sql = """
            with max_id_data as (
                select max(opsrecordid) as opsrecordid
                FROM opsmanagement.opsversion  AA
                inner join opsmanagement.opsrecord BB on 1 = 1 
                    and AA.opsversionid = BB.opsversion
                    and BB.state = 'FINISH'
                    and BB.resultjson like '%[:DatabaseFunction']%'
                WHERE 1 = 1 
                    AND AA.product = '[:DatabaseProduct]'
                    AND AA.project = '[:DatabaseProject]'
                    AND AA.opsversion = '[:DatabaseOPSVersion']' 
            ) select BB.resultjson::json -> '[:DatabaseFunction']' as resultjsonstr
            FROM opsmanagement.opsversion  AA
            inner join opsmanagement.opsrecord BB on 1 = 1 
                and AA.opsversionid = BB.opsversion
                --AND BB.opsrecordid in (select * FROM max_id_data)
                [:OPSRecordSQLCode] --AND BB.opsrecordid = 640
                and BB.state = 'FINISH'
            WHERE 1 = 1 
                AND AA.product = '[:DatabaseProduct]'
                AND AA.project = '[:DatabaseProject]'
                AND AA.opsversion = '[:DatabaseOPSVersion']' 
        """.replace("[:DatabaseProduct]", databaseProduct) \
            .replace("[:DatabaseProject]", databaseProject) \
            .replace("[:DatabaseOPSVersion']", databaseOPSVersion) \
            .replace("[:DatabaseFunction']", databaseFunction)
        if type(databaseOPSRecord) == type(1):
            sql = sql.replace("[:OPSRecordSQLCode]", "AND BB.opsrecordid = [:DatabaseOPSRecord]").replace(
                "[:DatabaseOPSRecord]", str(databaseOPSRecord))
        else:
            sql = sql.replace("[:OPSRecordSQLCode]", "AND BB.opsrecordid in (select * FROM max_id_data)")
        resultJsonDF = postgresCtrl.searchSQL(sql)
        databaseResultJson = resultJsonDF['resultjsonstr'][0]
        return databaseResultJson