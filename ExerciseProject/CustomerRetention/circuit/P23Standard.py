import os 
import pandas
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl

class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        sqlStrs = """
        DELETE FROM observationdata.standarddata AA 
        WHERE 1 = 1 
            AND AA.product = 'ExerciseProject'
            AND AA.project = 'CustomerRetention'
            AND AA.tablename = 'S0_0_1'
            AND AA.dt = '20220101' ; 
        INSERT INTO observationdata.standarddata
        SELECT
            product,'CustomerRetention' as project,tablename,dt
            , common_001,common_002,common_003,common_004,common_005
            , common_006,common_007,common_008,common_009,common_010
            , string_001,string_002,string_003,string_004,string_005
            , string_006,string_007,string_008,string_009,string_010
            , integer_001,integer_002,integer_003,integer_004,integer_005
            , integer_006,integer_007,integer_008,integer_009,integer_010
            , double_001,double_002,double_003,double_004,double_005
            , double_006,double_007,double_008,double_009,double_010
            , time_001,time_002,json_001,json_002
        FROM observationdata.standarddata AA
        WHERE 1 = 1
            AND AA.product = 'ExerciseProject'
            AND AA.project = 'RPM'
            AND AA.tablename = 'S0_0_1'
            AND AA.dt = '20220101' ; 
        """
        sqlStrArr = sqlStrs.split(";")[:-1]
        for sqlStr in sqlStrArr:
            postgresCtrl.executeSQL(sqlStr)
        return {"result": "OK"}, {}