import copy
import os
import datetime
from dotenv import load_dotenv
from package.common.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class RawDataFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["ResultArr"] :
                resultDict[key] = functionVersionInfo[key]
        if "GetXYData" in functionVersionInfo['FunctionType']:
            if functionVersionInfo['FunctionType'] == "GetXYData":
                otherInfo = self.rdGetXYData(functionVersionInfo)
            elif functionVersionInfo['FunctionType'] == "GetXYDataByFunctionRusult":
                otherInfo = self.rdGetXYDataByFunctionRusult(functionVersionInfo)
            elif functionVersionInfo['FunctionType'] == "GetXYDataByDatabaseRusult":
                otherInfo = self.rdGetXYDataByDatabaseRusult(functionVersionInfo)
            resultDict['FunctionItemType'] = otherInfo["FunctionItemType"]
            resultDict['MakeDataKeys'] = otherInfo['MakeDataKeys']
            resultDict['MakeDataInfo'] = otherInfo['MakeDataInfo']
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "GetSQLData":
            otherInfo = self.rdGetSQLData(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
            resultDict["SQLStrs"] = ""
        elif functionVersionInfo['FunctionType'] == "ExeSQLStrs":
            otherInfo = self.rdExeSQLStrs(functionVersionInfo)
            resultDict["SQLStrs"] = ""
        elif functionVersionInfo['FunctionType'] == "MakeTagText":
            otherInfo = self.rdMakeTagText(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    # fvInfo -- functionVersionInfo

    @classmethod
    def rdGetXYData(self, fvInfo):
        otherInfo = {}
        fvInfo = self.makeIntactFVInfoByXYData(fvInfo)
        if fvInfo['FunctionItemType'] == "ByDT":
            otherInfo["AnalysisDataInfoDF"],otherInfo["AnalysisDataCountDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
            otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        elif fvInfo['FunctionItemType'] == "ByDTDiff":
            otherInfo["AnalysisDataInfoDF"],otherInfo["AnalysisDataCountDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
            otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        elif fvInfo['FunctionItemType'] == "ByDTDiffFromDF":
            otherInfo["AnalysisDataInfoDF"],otherInfo["AnalysisDataCountDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
            otherInfo["DFArr"] = self.makeTagDataDFArrByDF(fvInfo, otherInfo)
        fvInfo = self.makeClearFVInfoByXYData(fvInfo)
        otherInfo['FunctionItemType'] = fvInfo["FunctionItemType"]
        otherInfo['MakeDataKeys'] = fvInfo['MakeDataKeys']
        otherInfo['MakeDataInfo'] = self.makeAnalysisDataInfoToMakeDataInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdGetXYDataByFunctionRusult(self, fvInfo):
        otherInfo = {}
        fvInfo = self.makeIntactFVInfoByXYData(fvInfo)
        otherInfo["AnalysisDataInfoDF"],otherInfo["AnalysisDataCountDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        fvInfo = self.makeClearFVInfoByXYData(fvInfo)
        otherInfo['FunctionItemType'] = fvInfo["FunctionItemType"]
        otherInfo['MakeDataKeys'] = fvInfo['MakeDataKeys']
        otherInfo['MakeDataInfo'] = self.makeAnalysisDataInfoToMakeDataInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdGetXYDataByDatabaseRusult(self, fvInfo):
        otherInfo = {}
        databaseResultJson = self.getDatabaseResultJson(fvInfo)
        fvInfo["FunctionItemType"] = databaseResultJson["FunctionItemType"]
        fvInfo["MakeDataKeys"] = databaseResultJson["MakeDataKeys"]
        fvInfo["MakeDataInfo"] = databaseResultJson["MakeDataInfo"]
        fvInfo = self.makeIntactFVInfoByXYData(fvInfo)
        otherInfo["AnalysisDataInfoDF"],otherInfo["AnalysisDataCountDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        fvInfo = self.makeClearFVInfoByXYData(fvInfo)
        otherInfo['FunctionItemType'] = fvInfo["FunctionItemType"]
        otherInfo['MakeDataKeys'] = fvInfo['MakeDataKeys']
        otherInfo['MakeDataInfo'] = self.makeAnalysisDataInfoToMakeDataInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdGetSQLData(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.makeGetSQLStrsByDataBase(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdExeSQLStrs(self, fvInfo):
        otherInfo = {}
        self.makeExeSQLStrsByDataBase(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdMakeTagText(self, fvInfo):
        otherInfo = {}
        self.makeTagTextByDataBase(fvInfo, otherInfo)
        return otherInfo

    # ==================================================   getXYData  ==================================================

    @classmethod
    def makeIntactFVInfoByXYData(self, fvInfo):
        import re
        def getGFuncDict (): # ,"avg":"avg({})",
            # 相關gFunc的Aggregate Functions用法可以參閱 https://www.postgresql.org/docs/15/functions-aggregate.html 這個網址
            return {"sum" : "sum({})","count":"count({})","max":"max({})","min":"min({})"}

        fvInfo['FunctionType'] = "GetXYData" if "FunctionType" not in fvInfo.keys() else fvInfo['FunctionType']
        fvInfo['FunctionItemType'] = "ByDTDiff" if "FunctionItemType" not in fvInfo.keys() else fvInfo['FunctionItemType']
        fvInfo['MakeDataKeys'] = ["common_001"] if "MakeDataKeys" not in fvInfo.keys() else fvInfo['MakeDataKeys']
        if "DataTime" not in fvInfo.keys() and "DTDiff" in fvInfo['FunctionItemType'] :
            raise Exception("DataTime is not in fvInfo")
        for makeDataInfo in fvInfo['MakeDataInfo']:
            makeDataInfo['DataType'] = 'X' if 'DataType' not in makeDataInfo.keys() else makeDataInfo['DataType']
            if 'Product' not in makeDataInfo.keys() :
                raise Exception("Product is not in MakeDataInfo")
            if 'Project' not in makeDataInfo.keys() :
                raise Exception("Project is not in MakeDataInfo")
            if 'Version' not in makeDataInfo.keys() :
                raise Exception("Version is not in MakeDataInfo")
            if 'ColumnNumbers' not in makeDataInfo.keys():
                raise Exception("ColumnNumbers is not in MakeDataInfo")
            if fvInfo['FunctionItemType'] == "ByDT":
                if 'DT' not in makeDataInfo.keys():
                    raise Exception("DT is not in MakeDataInfo")
                makeDataInfo['DTStr']  = "{}".format(makeDataInfo["DT"].replace("-", ""))
                makeDataInfo['DTSQL']  = "'{}'".format(makeDataInfo["DT"].replace("-", ""))
                makeDataInfo['DTNameStr'] = "{}".format(makeDataInfo["DT"].replace("-", ""))
            elif fvInfo['FunctionItemType'] == "ByDTDiff" or fvInfo['FunctionItemType'] == "ByDTDiffFromDF":
                if 'DTDiff' not in makeDataInfo.keys():
                    raise Exception("DTDiff is not in MakeDataInfo")
                makeDataInfo['DTStr'] = (datetime.datetime.strptime(fvInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=makeDataInfo["DTDiff"])).strftime("%Y%m%d")
                makeDataInfo['DTSQL'] = "to_char((date '{}' + integer '{}'),'yyyyMMdd')".format(fvInfo["DataTime"],str(makeDataInfo["DTDiff"]))
                makeDataInfo['DTNameStr'] = "p" + str(abs(makeDataInfo["DTDiff"])) if makeDataInfo["DTDiff"] >= 0 else "n" + str(abs(makeDataInfo["DTDiff"]))
            if 'GFunc' not in makeDataInfo.keys() and 'GFuncSQL' not in makeDataInfo.keys():
                makeDataInfo['GFunc'] = "sum"
                makeDataInfo['GFuncSQL'] = getGFuncDict()[makeDataInfo['GFunc']]
            elif 'GFuncSQL' in makeDataInfo.keys() :
                makeDataInfo['GFunc'] = str.lower(re.sub(r'[^a-zA-Z0-9_]', '', makeDataInfo['GFuncSQL']))
            elif makeDataInfo['GFunc'] in getGFuncDict().keys() :
                makeDataInfo['GFuncSQL'] = getGFuncDict()[makeDataInfo['GFunc']]
            else :
                raise Exception("GFunc or GFuncSQL is Error")
        return fvInfo

    @classmethod
    def makeClearFVInfoByXYData(self, fvInfo):
        for makeDataInfo in fvInfo['MakeDataInfo']:
            # if 'DTStr' in makeDataInfo.keys():
            #     del makeDataInfo['DTSQL']
            if 'DTSQL' in makeDataInfo.keys():
                del makeDataInfo['DTSQL']
            # if 'DTNameStr' in makeDataInfo.keys():
            #     del makeDataInfo['DTNameStr']
        return fvInfo

    @classmethod
    def makeAnalysisDataInfoDFByDataInfo(self,fvInfo,otherInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        doubleColumnNameArr = self.getDoubleColumnArr()

        makeDataDateStr = fvInfo["DataTime"]
        makeDataKeyArr = fvInfo["MakeDataKeys"]
        makeMaxCloumnCount = fvInfo["MakeMaxColumnCount"] if "MakeMaxColumnCount" in fvInfo.keys() else 9999999999
        makeDataInfoArr = fvInfo["MakeDataInfo"]

        countColumnsSQL = ""
        for columnName in doubleColumnNameArr:
            countColumnSQL = "\n                    , SUM(CASE WHEN AA.{} is not null then 1 else 0 end ) as {}"
            countColumnSQL = countColumnSQL.format(columnName, columnName)
            countColumnsSQL = countColumnsSQL + countColumnSQL

        countWheresSQL = ""
        for makeDataInfo in makeDataInfoArr:
            countWhereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = {} )"
            countWhereSQL = countWhereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"], makeDataInfo["Version"],makeDataInfo["DTSQL"])
            countWheresSQL = countWheresSQL + countWhereSQL

        infoWheresSQL = ""
        for makeDataInfo in makeDataInfoArr:
            infoWhereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = {} )"
            infoWhereSQL = infoWhereSQL.format(
                makeDataInfo["Product"],
                makeDataInfo["Project"],
                makeDataInfo['Version'].split("_")[0] + '_' + makeDataInfo['Version'].split("_")[1] + '_0',
                makeDataInfo["DTSQL"]
            )
            infoWheresSQL = infoWheresSQL + infoWhereSQL

        infoSQL = """
            SELECT
                AA.product as product
                , AA.project as project
                , AA.common_015::json->> 'DataVersion' as version
                , AA.dt as dt
                , AA.common_011 as enname
                , AA.common_012 as cnname
                , AA.common_013 as dataindex
                , AA.common_014 as memo
                , AA.common_015::json as messageinfo
            FROM observationdata.analysisdata AA
            where 1 = 1
                AND ( 1 != 1 [:WheresSQL] 
                )
        """.replace("[:WheresSQL]", infoWheresSQL)
        analysisDataInfoDF = postgresCtrl.searchSQL(infoSQL)
        countSQL = """
            SELECT
                AA.product as product
                , AA.project as project
                , AA.version as version
                , AA.dt as dt
                , AA.common_013 as dataindex [:ColumnsSQL] 
            FROM observationdata.analysisdata AA
            where 1 = 1
                AND ( 1 != 1 [:WheresSQL] 
                )
            GROUP BY
                AA.product
                , AA.project
                , AA.version
                , AA.dt
                , AA.common_013
        """.replace("[:ColumnsSQL]", countColumnsSQL).replace("[:WheresSQL]", countWheresSQL)

        analysisDataCountDF = postgresCtrl.searchSQL(countSQL)
        return analysisDataInfoDF , analysisDataCountDF

    @classmethod
    def makeAnalysisDataInfoToMakeDataInfo(self, fvInfo, otherInfo):
        makeDataInfos = copy.deepcopy(fvInfo['MakeDataInfo'])
        for makeDataInfo in makeDataInfos:
            columnInfo = {}
            for dataIndex, dataRow in otherInfo["AnalysisDataInfoDF"].iterrows():
                if (dataRow["product"] != makeDataInfo["Product"]) \
                        | (dataRow["project"] != makeDataInfo["Project"]) \
                        | (dataRow["project"] != makeDataInfo["Project"]) \
                        | (dataRow["dt"] != makeDataInfo["DTStr"]) \
                        | (dataRow["version"] != makeDataInfo["Version"]):
                    continue
                if int(dataRow["dataindex"]) not in makeDataInfo["ColumnNumbers"] :
                    continue
                columnInfo[dataRow["dataindex"]] = {
                    "enname": dataRow["enname"],
                    "cnname": dataRow["cnname"],
                    "memo": dataRow["memo"],
                    "messageinfo": dataRow["messageinfo"],
                }
            makeDataInfo["ColumnInfo"] = columnInfo
        return makeDataInfos

    @classmethod
    def makeTagDataDFArrByDataInfo(self,fvInfo,otherInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        def makeSingleTagDataSQL(dataInfo,otherInfo):
            infoKeysSQL = otherInfo["InfoKeysSQL"]
            yColumnsSQL = otherInfo["YColumnsSQL"]
            xColumnsSQL = otherInfo["XColumnsSQL"]
            wheresSQL = otherInfo["WheresSQL"]
            groupKeysSQL = otherInfo["GroupKeysSQL"]
            havingsSQL = otherInfo["HavingsSQL"]
            sql = """
                SELECT [:CommonKeySQL] [:YColumnsSQL] [:XColumnsSQL] 
                FROM observationdata.analysisdata AA
                where 1 = 1
                    AND ( 1 != 1 [:WheresSQL] 
                    )
                GROUP BY [:GroupKeySQL]
                HAVING SUM(1) = SUM(1) [:HavingsSQL]
                ORDER BY [:GroupKeySQL]
            """
            sql = sql.replace("[:CommonKeySQL]", infoKeysSQL)
            sql = sql.replace("[:YColumnsSQL]", yColumnsSQL)
            sql = sql.replace("[:XColumnsSQL]", xColumnsSQL)
            sql = sql.replace("[:WheresSQL]", wheresSQL)
            sql = sql.replace("[:GroupKeySQL]", groupKeysSQL)
            sql = sql.replace("[:HavingsSQL]", havingsSQL)
            return sql

        doubleColumnNameArr = self.getDoubleColumnArr()

        makeDataDateStr = fvInfo["DataTime"]
        makeDataKeyArr = fvInfo["MakeDataKeys"]
        makeDataInfoArr = fvInfo["MakeDataInfo"]
        analysisDataInfoDF = otherInfo["AnalysisDataCountDF"]

        infoKeysSQL = ""
        yColumnsSQL = ""
        xColumnsSQL = ""
        wheresSQL = ""
        groupKeysSQL = ""
        havingsSQL = ""

        for dataKey in makeDataKeyArr:
            infoKeySQL = "\n                    AA.{} as {}".format(dataKey,dataKey) if infoKeysSQL == "" else "\n                    , AA.{} as {}".format(dataKey, dataKey)
            groupKeySQL = "\n                    AA.{}".format(dataKey) if groupKeysSQL == "" else "\n                    , AA.{}".format(dataKey)
            infoKeysSQL = infoKeysSQL + infoKeySQL
            groupKeysSQL = groupKeysSQL + groupKeySQL

        for makeDataInfo in makeDataInfoArr:
            makeDataInfo["MakeDataDateStr"] = makeDataDateStr
            whereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = {} )"
            whereSQL = whereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"],makeDataInfo["Version"], makeDataInfo["DTSQL"])
            wheresSQL = wheresSQL + whereSQL

        sqlInfoArr = []

        sqlInfo = {}
        for makeDataInfo in makeDataInfoArr:
            for dataIndex, dataRow in analysisDataInfoDF.iterrows():
                if (dataRow["product"] != makeDataInfo["Product"]) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | (dataRow["dt"] != makeDataInfo["DTStr"]) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | ( dataRow["version"] != makeDataInfo["Version"]) :
                    continue
                columnNumberArr = makeDataInfo["ColumnNumbers"] if "ColumnNumbers" in makeDataInfo.keys() else []
                for columnName in doubleColumnNameArr:
                    haveDataindex = True if dataRow["dataindex"] != None else False
                    dataindex = int(dataRow["dataindex"]) if dataRow["dataindex"] != None else 1
                    columnindex = int(columnName.split("_")[1])
                    columnNumber = (dataindex - 1) * 200 + columnindex
                    if columnNumberArr != [] and columnNumber not in columnNumberArr:
                        continue
                    if dataRow[columnName] > 0 :
                        columnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(dataRow["product"], dataRow["project"], makeDataInfo['DTNameStr'], str(columnNumber), makeDataInfo['GFunc'],dataRow["version"]))
                        if haveDataindex :
                            gFuncSQL = makeDataInfo['GFuncSQL'].format("CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' AND AA.commondata_013 ='{}' then AA.{} else null end")
                            gFuncSQL = gFuncSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], dataRow["dataindex"], columnName)
                        else :
                            gFuncSQL = makeDataInfo['GFuncSQL'].format("CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' then AA.{} else null end")
                            gFuncSQL = gFuncSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], columnName)
                        columnSQL = "\n                    , {} as {}".format(gFuncSQL, columnFullName)
                        if makeDataInfo['DataType'] == "Y":
                            yColumnsSQL = yColumnsSQL + columnSQL
                        elif makeDataInfo['DataType'] == "X":
                            xColumnsSQL = xColumnsSQL + columnSQL
                        elif makeDataInfo['DataType'] == "Filter":
                            havingSQLArr = makeDataInfo["HavingSQL"]
                            if columnNumber in columnNumberArr :
                                havingsSQL += "\n                    AND {} {}".format(gFuncSQL, havingSQLArr[columnNumberArr.index(columnNumber)])

        sqlInfo["InfoKeysSQL"] = infoKeysSQL
        sqlInfo["YColumnsSQL"] = yColumnsSQL
        sqlInfo["XColumnsSQL"] = xColumnsSQL
        sqlInfo["WheresSQL"] = wheresSQL
        sqlInfo["GroupKeysSQL"] = groupKeysSQL
        sqlInfo["HavingsSQL"] = havingsSQL
        sqlInfoArr.append(sqlInfo)
        dfArr = []
        for sqlInfo in sqlInfoArr :
            sql = makeSingleTagDataSQL(fvInfo,sqlInfo)
            df = postgresCtrl.searchSQL(sql)
            dfArr.append(df)
        return dfArr

    @classmethod
    def makeTagDataDFArrByDF(self,fvInfo,otherInfo):
        # 指定DF做相關的欄位分拆
        import pandas

        makeDataDateStr = fvInfo["DataTime"]
        makeDataKeyArr = fvInfo["MakeDataKeys"]
        makeDataInfoArr = fvInfo["MakeDataInfo"]

        oriRawDataDFArr = fvInfo["ResultArr"]
        rawDataDFArr = []

        for oriRawDataDF in oriRawDataDFArr :
            rawDataDF = pandas.DataFrame()
            for makeDataKey in makeDataKeyArr:
                rawDataDF[makeDataKey] = oriRawDataDF[makeDataKey]
            for makeDataInfo in makeDataInfoArr:
                columnNumberArr = makeDataInfo["ColumnNumbers"]
                for columnNumber in columnNumberArr:
                    oriDTStr = (datetime.datetime.strptime(makeDataDateStr, "%Y-%m-%d") + datetime.timedelta(days=makeDataInfo["DTDiff"])).strftime("%Y%m%d")
                    newDTStr = "p" + str(abs(makeDataInfo["DTDiff"])) if makeDataInfo["DTDiff"] >= 0 else "n" + str(abs(makeDataInfo["DTDiff"]))
                    oriColumnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(makeDataInfo["Product"], makeDataInfo["Project"], oriDTStr,str(columnNumber),makeDataInfo["GFunc"], makeDataInfo["Version"]))
                    newColumnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(makeDataInfo["Product"], makeDataInfo["Project"], newDTStr,str(columnNumber),makeDataInfo["GFunc"], makeDataInfo["Version"]))
                    rawDataDF[newColumnFullName] = oriRawDataDF[oriColumnFullName]
            rawDataDFArr.append(rawDataDF)

        return rawDataDFArr

    # ==================================================  ExeSQLStrs ==================================================

    @classmethod
    def makeGetSQLStrsByDataBase(self, fvInfo, otherInfo):
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
        dfArr = []
        for sqlStr in sqlStrArr:
            dfArr.append(postgresCtrl.searchSQL(sqlStr))
        return dfArr

    # ==================================================  MakeTagText ==================================================

    @classmethod
    def makeTagTextByDataBase(self,fvInfo,otherInfo):
        from package.artificialintelligence.common.virtualentity.TagText import TagText
        tagText = TagText()
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        tagText.setFeatureDictByFilePath(filePath=fvInfo["FilePath"])
        if fvInfo["FeatureType"] == "General":
            tagText.makeFeatureDFByGeneralFD()
        insertDataDF = tagText.getFeatureDF()
        insertDataDF['product'] = fvInfo["Product"]
        insertDataDF['project'] = fvInfo["Project"]
        insertDataDF['version'] = fvInfo["Version"]
        insertDataDF['dt'] = fvInfo["DataTime"].replace("-","")
        tableFullName = "observationdata.analysisdata"
        for column in self.getAnalysisColumnNameArr():
            if column not in insertDataDF.columns:
                insertDataDF[column] = None
        insertTableInfoDF = postgresCtrl.getTableInfoDF(tableFullName)
        deleteSQL = """
            DELETE FROM observationdata.analysisdata
            WHERE 1 = 1 
                AND product = '{}'
                AND project = '{}'
                AND version = '{}'
                AND dt = '{}'
        """.format(fvInfo["Product"], fvInfo["Project"], fvInfo["Version"], fvInfo["DataTime"].replace("-",""))
        postgresCtrl.executeSQL(deleteSQL)
        postgresCtrl.insertDataList(tableFullName, insertTableInfoDF, insertDataDF)
        return {}