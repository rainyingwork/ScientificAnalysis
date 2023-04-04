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
        if functionVersionInfo['FunctionType'] == "GetXYData":
            otherInfo = self.rdGetXYData(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "GetXYDataByFunctionRusult":
            otherInfo = self.rdGetXYDataByFunctionRusult(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "GetXYDataByDatabaseRusult":
            otherInfo = self.rdGetXYDataByDatabaseRusult(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
            resultDict['MakeDataKeys'], resultDict['MakeDataInfo'] = otherInfo['MakeDataKeys'], otherInfo["MakeDataInfo"]
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
        if "FunctionItemType" not in fvInfo.keys(): # 預設為ByDTDiff
            fvInfo['FunctionItemType'] = "ByDTDiff"
        if fvInfo['FunctionItemType'] == "ByDT":
            otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
            otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        elif fvInfo['FunctionItemType'] == "ByDTDiff":
            otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
            otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        elif fvInfo['FunctionItemType'] == "ByDTDiffFromDF":
            otherInfo["DFArr"] = self.makeTagDataDFArrByDF(fvInfo, otherInfo)


        return otherInfo

    @classmethod
    def rdGetXYDataByFunctionRusult(self, fvInfo):
        otherInfo = {}
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdGetXYDataByDatabaseRusult(self, fvInfo):
        otherInfo = {}
        databaseResultJson = self.getDatabaseResultJson(fvInfo)
        fvInfo["MakeDataKeys"] = databaseResultJson["MakeDataKeys"]
        fvInfo["MakeDataInfo"] = databaseResultJson["MakeDataInfo"]
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        otherInfo["MakeDataKeys"], otherInfo["MakeDataInfo"] = fvInfo["MakeDataKeys"] , fvInfo["MakeDataInfo"]
        return otherInfo

    @classmethod
    def rdGetSQLData(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.makeGetSQLStrsByDataBase(fvInfo ,otherInfo)
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

        infoColumnsSQL = ""
        for columnName in doubleColumnNameArr:
            infoColumnSQL = "\n                    , SUM(CASE WHEN AA.{} is not null then 1 else 0 end ) as {}"
            infoColumnSQL = infoColumnSQL.format(columnName, columnName)
            infoColumnsSQL = infoColumnsSQL + infoColumnSQL

        infoWheresSQL = ""
        for makeDataInfo in makeDataInfoArr:
            makeDataInfo["MakeDataDateStr"] = makeDataDateStr
            if fvInfo['FunctionItemType'] == "ByDT":
                dtStr = "'{}'".format(makeDataInfo["DT"].replace("-",""))
            elif fvInfo['FunctionItemType'] == "ByDTDiff":
                dtStr = "to_char((date '{}' + integer '{}'),'yyyyMMdd')".format(makeDataInfo["MakeDataDateStr"],str(makeDataInfo["DTDiff"]))
            infoWhereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = {} )"
            infoWhereSQL = infoWhereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"], makeDataInfo["Version"],dtStr)
            infoWheresSQL = infoWheresSQL + infoWhereSQL

        infoSQL = """
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
        """.replace("[:ColumnsSQL]", infoColumnsSQL).replace("[:WheresSQL]", infoWheresSQL)

        analysisDataInfoDF = postgresCtrl.searchSQL(infoSQL)
        return analysisDataInfoDF

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
        analysisDataInfoDF = otherInfo["AnalysisDataInfoDF"]

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
            if fvInfo['FunctionItemType'] == "ByDT":
                dtStr = "'{}'".format(makeDataInfo["DT"].replace("-", ""))
            elif fvInfo['FunctionItemType'] == "ByDTDiff":
                dtStr = "to_char((date '{}' + integer '{}'),'yyyyMMdd')".format(makeDataInfo["MakeDataDateStr"],str(makeDataInfo["DTDiff"]))
            whereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = {} )"
            whereSQL = whereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"],makeDataInfo["Version"], dtStr)
            wheresSQL = wheresSQL + whereSQL

        sqlInfo = {}
        sqlInfoArr = []
        signleCloumnCount = 0

        sqlInfo = {}
        for makeDataInfo in makeDataInfoArr:
            for dataIndex, dataRow in analysisDataInfoDF.iterrows():
                if fvInfo['FunctionItemType'] == "ByDT":
                    dtStr = "{}".format(makeDataInfo["DT"].replace("-", ""))
                elif fvInfo['FunctionItemType'] == "ByDTDiff":
                    dtStr = (datetime.datetime.strptime(makeDataDateStr, "%Y-%m-%d") + datetime.timedelta(days=makeDataInfo["DTDiff"])).strftime("%Y%m%d")
                if (dataRow["product"] != makeDataInfo["Product"]) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | (dataRow["dt"] != dtStr) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | ( dataRow["version"] != makeDataInfo["Version"]) :
                    continue
                isNoneDrop = makeDataInfo["IsNoneDrop"] if "IsNoneDrop" in makeDataInfo.keys() else True
                datatype = makeDataInfo["DataType"] if "DataType" in makeDataInfo.keys() else "X"
                if fvInfo['FunctionItemType'] == "ByDT":
                    dtNameStr = "{}".format(makeDataInfo["DT"].replace("-", ""))
                elif fvInfo['FunctionItemType'] == "ByDTDiff":
                    dtNameStr = "p" + str(abs(makeDataInfo["DTDiff"])) if makeDataInfo["DTDiff"] >= 0 else "n" + str(abs(makeDataInfo["DTDiff"]))
                columnNumberArr = makeDataInfo["ColumnNumbers"] if "ColumnNumbers" in makeDataInfo.keys() else []
                for columnName in doubleColumnNameArr:
                    sumSQL = ""
                    columnSQL = ""
                    haveDataindex = True if dataRow["dataindex"] != None else False
                    dataindex = int(dataRow["dataindex"]) if dataRow["dataindex"] != None else 1
                    columnindex = int(columnName.split("_")[1])
                    columnNumber = (dataindex - 1) * 200 + columnindex
                    if columnNumberArr != [] and columnNumber not in columnNumberArr:
                        continue
                    if dataRow[columnName] > 0 or isNoneDrop == False:
                        columnFullName = "{}_{}_{}_{}_{}".format(dataRow["product"], dataRow["project"], dtNameStr, str(columnNumber),dataRow["version"])
                        if haveDataindex :
                            sumSQL = "SUM(CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' AND AA.commondata_013 ='{}' then AA.{} else null end )"
                            sumSQL = sumSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], dataRow["dataindex"], columnName)
                            columnSQL = "\n                    , {} as {}".format(sumSQL, columnFullName)
                        else :
                            sumSQL = "SUM(CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' then AA.{} else null end )"
                            sumSQL = sumSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], columnName)
                            columnSQL = "\n                    , {} as {}".format(sumSQL, columnFullName)

                        if datatype == "Y":
                            yColumnsSQL = yColumnsSQL + columnSQL
                        elif datatype == "X":
                            xColumnsSQL = xColumnsSQL + columnSQL
                        elif datatype == "Filter":
                            havingSQLArr = makeDataInfo["HavingSQL"]
                            if columnNumber in columnNumberArr :
                                havingsSQL += "\n                    AND {} {}".format(sumSQL, havingSQLArr[columnNumberArr.index(columnNumber)])

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
                    oriColumnFullName = str.lower("{}_{}_{}_{}_{}".format(makeDataInfo["Product"], makeDataInfo["Project"], oriDTStr,str(columnNumber), makeDataInfo["Version"]))
                    newColumnFullName = str.lower("{}_{}_{}_{}_{}".format(makeDataInfo["Product"], makeDataInfo["Project"], newDTStr,str(columnNumber), makeDataInfo["Version"]))
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