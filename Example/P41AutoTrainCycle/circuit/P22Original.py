
class Original():

    @classmethod
    def O0_0_1(self, functionInfo):
        import os
        import pandas
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        postgresCtrl.executeSQL("DELETE FROM originaldata.archive_countryinitialism")
        insertDataDF = pandas.read_csv("CustomerAnalysis/P2301Archive/file/data/customers.csv")
        insertDataDF.columns = [
            'archiveid','gender','age','numberofkids'
        ]
        tableName = "originaldata.archive_countryinitialism"
        tableInfoDF = postgresCtrl.getTableInfoDF(tableName)
        postgresCtrl.insertDataByIO(tableName,tableInfoDF,insertDataDF)
        return {"result": "OK"}, {}

    @classmethod
    def O0_0_2(self, functionInfo):
        import os
        import pandas
        from datetime import datetime
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        postgresCtrl.executeSQL("DELETE FROM originaldata.archive_pings")
        insertDataDF = pandas.read_csv("CustomerAnalysis/P2301Archive/file/data/pings.csv")
        insertDataDF.columns = [
            'archiveid','pingtime'
        ]
        insertDataDF["pingtime"] = insertDataDF["pingtime"].apply(lambda x: datetime.fromtimestamp(x))
        tableName = "originaldata.archive_pings"
        tableInfoDF = postgresCtrl.getTableInfoDF(tableName)
        postgresCtrl.insertDataByIO(tableName,tableInfoDF,insertDataDF)
        return {"result": "OK"}, {}

    @classmethod
    def O0_0_3(self, functionInfo):
        import os
        import pandas
        from datetime import datetime
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        postgresCtrl.executeSQL("DELETE FROM originaldata.archive_test")
        insertDataDF = pandas.read_csv("CustomerAnalysis/P2301Archive/file/data/test.csv")
        insertDataDF.columns = [
            'archiveid','archivedate','onlinehours'
        ]
        insertDataDF["archivedate"] = insertDataDF["archivedate"].apply(lambda x: datetime.strptime(x,"%d/%m/%y"))
        tableName = "originaldata.archive_test"
        tableInfoDF = postgresCtrl.getTableInfoDF(tableName)
        postgresCtrl.insertDataByIO(tableName,tableInfoDF,insertDataDF)
        return {"result": "OK"}, {}

