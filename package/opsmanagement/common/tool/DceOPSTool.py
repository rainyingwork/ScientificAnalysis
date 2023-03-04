import os
from package.common.common.database.PostgresCtrl import PostgresCtrl

class DceOPSTool () :

    def __init__(self):
        self.postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

    def getRunDCEOPSDF(self , runType , batchNumber ):
        sql = """
            select 
                BB.product 
                , BB.project 
                , BB.opsversion 
                , AA.opsrecordid 
            from opsmanagement.opsrecord AA
            inner join opsmanagement.opsversion BB on 1 = 1
                and AA.opsversion = BB.opsversionid 
            where 1 = 1 
                and AA.opsorderjson::json->>'RunType' =  '[:RunType]'
                and AA.opsorderjson::json->>'BatchNumber' =  '[:BatchNumber]'
                and AA.deletetime is null 
                and AA.state = 'RUN'
        """.replace("[:RunType]" , runType).replace("[:BatchNumber]" , batchNumber)
        df = self.postgresCtrl.searchSQL(sql)
        return df