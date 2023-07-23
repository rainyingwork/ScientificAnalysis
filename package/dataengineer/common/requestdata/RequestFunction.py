import os
from package.dataengineer.common.common.CommonFunction import CommonFunction

class RequestFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["ResultArr"] :
                resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "RequestByRestAPI":
            pass ; # otherInfo = self.reRequestByRequest(functionVersionInfo)
        elif functionVersionInfo['FunctionType'] == "RequestByChrome":
            pass ; # otherInfo = self.reRequestByChrome(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ==================================================              ==================================================
    @classmethod
    def reRequestByRestAPI(self, fvInfo):
        otherInfo = {}
        self.requestByRestAPI(fvInfo, otherInfo)
        return otherInfo

    # ==================================================              ==================================================

    @classmethod
    def requestByRestAPI(self, fvInfo, otherInfo):
        pass

    @classmethod
    def sendRequest(self, fvInfo , useType = 'Request'):
        import os
        import requests
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        if useType == 'Request':
            if fvInfo['RequestType'] == "GET":
                response = requests.get(fvInfo['RequestURL'] + '?' + '&'.join([f"{k}={v}" for k, v in fvInfo['RequestParameter'].items()]))
            elif fvInfo['RequestType'] == "Post":
                response = requests.post(fvInfo['RequestURL'], data=fvInfo['RequestParameter'])
            return response
        elif useType == "Chrome":
            chrome_options = webdriver.chrome.options.Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chromeDriverName = 'chromedriver.exe' if os.name == "nt" else 'chromedriver'
            driver = webdriver.Chrome(options=chrome_options)
            if fvInfo['RequestType'] == "GET":
                driver.get(fvInfo['RequestURL'])
            return driver


    # ==================================================              ==================================================

    @classmethod
    def insertRequestData(self, product, project, exefunction, startdate, enddate, title, requestDataDF , useType):
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

        deleteDataSQL = """
            UPDATE observationdata.requestdata SET deletetime = now() 
            WHERE 1 = 1 
                AND product = '{}' 
                AND project = '{}' 
                AND exefunction = '{}' 
                AND requesttitle = '{}' 
                AND startdate = '{}' 
                AND enddate = '{}' 
                AND deletetime IS NULL
        """.format(product, project, exefunction,title, startdate.strftime("%Y-%m-%d"), enddate.strftime("%Y-%m-%d"))

        postgresCtrl.executeSQL(deleteDataSQL)

        tableFullName = " observationdata.requestdata"

        for column in self.getRequestColumnNameArr():
            if column not in requestDataDF.columns:
                requestDataDF[column] = None
        requestDataDF = requestDataDF[self.getRequestColumnNameArr()]
        insertTableInfoDF = postgresCtrl.getTableInfoDF(tableFullName)

        if useType == 'SQL':
            postgresCtrl.insertDataList(tableFullName, insertTableInfoDF, requestDataDF)
        elif useType == 'IO':
            postgresCtrl.insertDataByIO(tableFullName, insertTableInfoDF, requestDataDF)

        return {'result':'OK'} , {}

    @classmethod
    def getRequestDataDF(self, product, project, exefunction, startdate, enddate):
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

        selectDataSQL = """
            SELECT * FROM observationdata.requestdata 
            WHERE 1 = 1 
                AND product = '{}' 
                AND project = '{}' 
                AND exefunction = '{}' 
                AND startdate >= '{}' 
                AND enddate <= '{}' 
                AND deletetime IS NULL
        """.format(product, project, exefunction, startdate.strftime("%Y-%m-%d"), enddate.strftime("%Y-%m-%d"))

        requestDataDF = postgresCtrl.searchSQL(selectDataSQL)

        return requestDataDF

    # ==================================================              ==================================================

    # ==================================================              ==================================================

    @classmethod
    def getRequestColumnNameArr(self):
        columnNameArr = [
            "createtime", "modifytime", "deletetime", "requestdataid"
            , "product", "project", "exefunction", "startdate", "enddate"
            , "requesturl", "requesttype", "requestparameter"
            , "requesttitle", "requestcontent"
        ]
        return columnNameArr




