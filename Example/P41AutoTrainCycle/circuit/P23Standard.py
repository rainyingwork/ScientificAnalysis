
class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        import os , copy
        import pandas
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["S0_0_1"])
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        sql = """
            DELETE FROM observationdata.standarddata 
            WHERE product = '[:Product]' AND project = '[:Project]' AND tablename = '[:TableName]' AND dt = '[:DateNoLine]' ; 
        
            Insert into observationdata.standarddata
            select distinct 
                '[:Product]' as product, '[:Project]' as project, '[:TableName]' as tablename, '[:DateNoLine]' as dt
                , archiveid::text as common_001, null as common_002, null as common_003, null as common_004, null as common_005
                , null as common_006, null as common_007, null as common_008, null as common_009, null as common_010
                , null as string_001, null as string_002, null as string_003, null as string_004, null as string_005
                , null as string_006, null as string_007, null as string_008, null as string_009, null as string_010
                , null::bigint as integer_001, null::bigint as integer_002
                , null::bigint as integer_003, null::bigint as integer_004
                , null::bigint as integer_005, null::bigint as integer_006
                , null::bigint as integer_007, null::bigint as integer_008
                , null::bigint as integer_009, null::bigint as integer_010
                , null::double precision as double_001, null::double precision as double_002
                , null::double precision as double_003, null::double precision as double_004
                , null::double precision as double_005, null::double precision as double_006
                , null::double precision as double_007, null::double precision as double_008
                , null::double precision as double_009, null::double precision as double_010
                , pingtime as time_001, pingtime + interval '15 second' as time_002
                , null as json_001, null as json_002
            from originaldata.archive_pings
            where 1 = 1
                and pingtime::date = '[:DateLine]' ; 
        """
        sql = sql.replace("[:Product]",functionInfo["Product"])
        sql = sql.replace("[:Project]",functionInfo["Project"])
        sql = sql.replace("[:TableName]","1102")
        sql = sql.replace("[:DateLine]",functionVersionInfo["DataTime"])
        sql = sql.replace("[:DateNoLine]",functionVersionInfo["DataTime"].replace("-",""))

        sqlStrArr = sql.split(";")[:-1]

        for sqlStr in sqlStrArr :
            postgresCtrl.executeSQL(sqlStr)

        return {"result": "OK"} , {}

