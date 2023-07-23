
class Crawler():

    @classmethod
    def C0_9_1(self, functionInfo):
        import os , copy
        import pandas , json
        import datetime
        from yahoo_fin import stock_info as si
        import yfinance as yf
        from package.dataengineer.common.requestdata.RequestFunction import RequestFunction
        from package.dataengineer.common.entity.RequestData import RequestData
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

        requestFunction = RequestFunction()
        requestData = RequestData()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["C0_9_1"])
        functionVersionInfo["Version"] = "C0_9_1"

        # 相關內容製作 ----------------------------------------------------------------------------------------------------

        stockIdArr = ["2616","2617","2618"]

        for stockId in stockIdArr:

            stockID = '{}.TW'.format(stockId)

            print("{} {}".format(stockID,functionVersionInfo["DataTime"]))

            ticker = yf.Ticker(stockID)
            historyDF = ticker.history(
                start="{}-01-01".format(functionVersionInfo["DataTime"][0:4])
                , end="{}-12-31".format(functionVersionInfo["DataTime"][0:4])
            ).reset_index()
            historyDF["Date"] = historyDF["Date"].astype(str)
            historyDict = historyDF.to_dict(orient='records')

            # 資料庫更新 ----------------------------------------------------------------------------------------------------
            insertValuesObjectArr = []
            insertValuesObject = {}
            insertValuesObject["createtime"] = datetime.datetime.now()
            insertValuesObject["modifytime"] = datetime.datetime.now()
            insertValuesObject["deletetime"] = None
            insertValuesObject["requestdataid"] = requestData.getNextPrimaryKeyId()
            insertValuesObject["product"] = functionInfo['Product']
            insertValuesObject["project"] = functionInfo['Project']
            insertValuesObject["exefunction"] = functionVersionInfo['Version']
            insertValuesObject["startdate"] = datetime.datetime.strptime(functionVersionInfo["DataTime"], "%Y-%m-%d")
            insertValuesObject["enddate"] = datetime.datetime.strptime(functionVersionInfo["DataTime"], "%Y-%m-%d")
            insertValuesObject["requesturl"] = "pipy://yfinance"
            insertValuesObject["requesttype"] = None
            insertValuesObject["requestparameter"] = json.dumps({"StockID":stockID}, ensure_ascii=False)
            insertValuesObject["requesttitle"] = "StockPriceByDay_{}".format(stockID)
            insertValuesObject["requestcontent"] = json.dumps(historyDict)
            insertValuesObjectArr.append(insertValuesObject)

            requestDataDF = pandas.DataFrame(insertValuesObjectArr)

            resultObject, globalObjectDict = requestFunction.insertRequestData(
                functionInfo['Product'], functionInfo['Project'], functionVersionInfo['Version']
                , insertValuesObject["startdate"] , insertValuesObject["enddate"]
                , "StockPriceByDay_{}".format(stockID) , requestDataDF, useType = "IO"
            )

            import time
            time.sleep(10)

        return resultObject, globalObjectDict
