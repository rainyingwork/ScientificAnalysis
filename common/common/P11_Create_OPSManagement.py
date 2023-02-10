import os
from package.common.common.database.tool.SqlTool import SqlTool
from package.common.common.database.PostgresCtrl import PostgresCtrl
from dotenv import load_dotenv
import time

sqlTool = SqlTool()

load_dotenv(dotenv_path="env/postgresql.env")
postgresCtrl = PostgresCtrl(
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_POST")),
    user=os.getenv("POSTGRES_USERNAME"),
    password=os.getenv("POSTGRES_PASSWORD"),
    database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"],
    schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"],
)

sqlFilePathArr = [
    # "common/common/file/init/P11_00_PG_CreateSchema_OPSManagement.sql",
    # "common/common/file/init/P11_01_PG_CreateTable_OPSManagement_OPSVersion.sql",
    # "common/common/file/init/P11_02_PG_CreateTable_OPSManagement_OPSRecord.sql",
    # "common/common/file/init/P11_02_PG_CreateTable_OPSManagement_OPSDetail.sql",
]

for sqlFilePath in sqlFilePathArr :
    if sqlFilePath == "" :
        continue
    sqlTool.loadSQLfile(sqlFilePath)
    sqlTool.replaceSQLStrs()
    sqlTool.makeSQLArrSplitSQLStr()
    sqlStrArr = sqlTool.getSQLArr()
    for sqlStr in sqlStrArr :
        time.sleep(0.5)
        postgresCtrl.executeSQL(sqlStr)
