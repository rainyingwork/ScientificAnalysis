import os
import time , datetime
import json
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class RawDataFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "GetXYData":
            otherInfo = self.rdGetXYData(functionVersionInfo)
            resultDict['ResultIDArr'] , globalObjectDict['ResultArr'] = otherInfo['DFIDArr'] , otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "GetXYDataByFunctionRusult":
            otherInfo = self.rdGetXYDataByFunctionRusult(functionVersionInfo)
            resultDict['ResultIDArr'], globalObjectDict['ResultArr'] = otherInfo['DFIDArr'], otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "GetXYDataByDatabaseRusult":
            otherInfo = self.rdGetXYDataByDatabaseRusult(functionVersionInfo)
            resultDict['ResultIDArr'], globalObjectDict['ResultArr'] = otherInfo['DFIDArr'], otherInfo["DFArr"]
            resultDict['MakeDataKeys'], resultDict['MakeDataInfo'] = otherInfo['MakeDataKeys'], otherInfo["MakeDataInfo"]
        elif functionVersionInfo['FunctionType'] == "GetSQLData":
            otherInfo = self.rdGetSQLData(functionVersionInfo)
        elif functionVersionInfo['FunctionType'] == "ExeSQLStrs":
            otherInfo = self.rdExeSQLStrs(functionVersionInfo)
        elif functionVersionInfo['FunctionType'] == "MakeTagText":
            otherInfo = self.rdMakeTagText(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================
    # fvInfo -- functionVersionInfo

    @classmethod
    def rdGetXYData(self, fvInfo):
        otherInfo = {}
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo ,otherInfo)
        otherInfo["DFIDArr"] , otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo,otherInfo)
        return otherInfo

    @classmethod
    def rdGetXYDataByFunctionRusult(self, fvInfo):
        otherInfo = {}
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFIDArr"], otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def rdGetXYDataByDatabaseRusult(self, fvInfo):
        otherInfo = {}
        databaseResultJson = self.getDatabaseResultJson(fvInfo)
        fvInfo["MakeDataKeys"] = databaseResultJson["MakeDataKeys"]
        fvInfo["MakeDataInfo"] = databaseResultJson["MakeDataInfo"]
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo, otherInfo)
        otherInfo["DFIDArr"], otherInfo["DFArr"] = self.makeTagDataDFArrByDataInfo(fvInfo, otherInfo)
        otherInfo["MakeDataKeys"], otherInfo["MakeDataInfo"] = fvInfo["MakeDataKeys"] , fvInfo["MakeDataInfo"]
        return otherInfo

    @classmethod
    def rdGetSQLData(self, fvInfo):
        otherInfo = {}
        otherInfo["AnalysisDataInfoDF"] = self.makeAnalysisDataInfoDFByDataInfo(fvInfo ,otherInfo)
        dfIDArr = self.makeTagDataDFArrByDataInfo(fvInfo,otherInfo)
        return dfIDArr

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
            , database="scientificanalysis"
            , schema="public"
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
            infoWhereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = to_char((date '{}' + integer '{}'),'yyyyMMdd'))"
            infoWhereSQL = infoWhereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"], makeDataInfo["Version"],makeDataInfo["MakeDataDateStr"], str(makeDataInfo["DTDiff"]))
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
            , database="scientificanalysis"
            , schema="public"
        )

        def makeSingleTagDataSQL(dataInfo,otherInfo):
            infoKeysSQL = otherInfo["InfoKeysSQL"]
            yColumnsSQL = otherInfo["YColumnsSQL"]
            xColumnsSQL = otherInfo["XColumnsSQL"]
            wheresSQL = otherInfo["WheresSQL"]
            groupKeysSQL = otherInfo["GroupKeysSQL"]
            sql = """
                SELECT [:CommonKeySQL] [:YColumnsSQL] [:XColumnsSQL] 
                FROM observationdata.analysisdata AA
                where 1 = 1
                    AND ( 1 != 1 [:WheresSQL] 
                    )
                GROUP BY [:GroupKeySQL]
                ORDER BY [:GroupKeySQL]
            """
            sql = sql.replace("[:CommonKeySQL]", infoKeysSQL)
            sql = sql.replace("[:YColumnsSQL]", yColumnsSQL)
            sql = sql.replace("[:XColumnsSQL]", xColumnsSQL)
            sql = sql.replace("[:WheresSQL]", wheresSQL)
            sql = sql.replace("[:GroupKeySQL]", groupKeysSQL)
            return sql

        doubleColumnNameArr = self.getDoubleColumnArr()

        makeDataDateStr = fvInfo["DataTime"]
        makeDataKeyArr = fvInfo["MakeDataKeys"]
        makeMaxCloumnCount = fvInfo["MakeMaxColumnCount"] if "MakeMaxColumnCount" in fvInfo.keys() else 9999999999
        makeDataInfoArr = fvInfo["MakeDataInfo"]
        analysisDataInfoDF = otherInfo["AnalysisDataInfoDF"]

        infoKeysSQL = ""
        groupKeysSQL = ""
        wheresSQL = ""
        yColumnsSQL = ""
        xColumnsSQL = ""

        for dataKey in makeDataKeyArr:
            infoKeySQL = "\n                    AA.{} as {}".format(dataKey,dataKey) if infoKeysSQL == "" else "\n                    , AA.{} as {}".format(dataKey, dataKey)
            groupKeySQL = "\n                    AA.{}".format(dataKey) if groupKeysSQL == "" else "\n                    , AA.{}".format(dataKey)
            infoKeysSQL = infoKeysSQL + infoKeySQL
            groupKeysSQL = groupKeysSQL + groupKeySQL

        for makeDataInfo in makeDataInfoArr:
            whereSQL = "\n                        OR (AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = to_char((date '{}' + integer '{}'),'yyyyMMdd'))"
            whereSQL = whereSQL.format(makeDataInfo["Product"], makeDataInfo["Project"], makeDataInfo["Version"],makeDataInfo["MakeDataDateStr"], str(makeDataInfo["DTDiff"]))
            wheresSQL = wheresSQL + whereSQL

        sqlInfo = {}
        sqlInfoArr = []
        signleCloumnCount = 0
        for dataIndex, dataRow in analysisDataInfoDF.iterrows():
            for makeDataInfo in makeDataInfoArr:
                dtStr = (datetime.datetime.strptime(makeDataDateStr, "%Y-%m-%d") + datetime.timedelta(days=makeDataInfo["DTDiff"])).strftime("%Y%m%d")
                if (dataRow["product"] != makeDataInfo["Product"]) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | (dataRow["dt"] != dtStr) \
                    | (dataRow["project"] != makeDataInfo["Project"]) \
                    | ( dataRow["version"] != makeDataInfo["Version"]) :
                    continue
                isNoneDrop = makeDataInfo["IsNoneDrop"] if "IsNoneDrop" in makeDataInfo.keys() else True
                datatype = makeDataInfo["DataType"]
                dtDiffStr = "p" + str(abs(makeDataInfo["DTDiff"])) if makeDataInfo["DTDiff"] >= 0 else "n" + str(abs(makeDataInfo["DTDiff"]))
                columnNumberArr = makeDataInfo["ColumnNumbers"] if "ColumnNumbers" in makeDataInfo.keys() else []
                for columnName in doubleColumnNameArr:
                    columnSQL = ""
                    haveDataindex = True if dataRow["dataindex"] != None else False
                    dataindex = int(dataRow["dataindex"]) if dataRow["dataindex"] != None else 1
                    columnindex = int(columnName.split("_")[1])
                    columnNumber = (dataindex - 1) * 200 + columnindex
                    if columnNumberArr != [] and columnNumber not in columnNumberArr:
                        continue
                    if dataRow[columnName] > 0 or isNoneDrop == False:
                        columnFullName = "{}_{}_{}_{}_{}".format(dataRow["product"], dataRow["project"], dtDiffStr, str(columnNumber),dataRow["version"])
                        if haveDataindex :
                            columnSQL = "\n                    , SUM(CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' AND AA.commondata_013 ='{}' then AA.{} else null end ) as {}"
                            columnSQL = columnSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], dataRow["dataindex"], columnName,columnFullName)
                        else :
                            columnSQL = "\n                    , SUM(CASE WHEN AA.product = '{}' AND AA.project='{}' AND AA.version='{}' AND AA.dt = '{}' then AA.{} else null end ) as {}"
                            columnSQL = columnSQL.format(dataRow["product"], dataRow["project"], dataRow["version"],dataRow["dt"], columnName,columnFullName)
                        if datatype == "Y":
                            yColumnsSQL = yColumnsSQL + columnSQL
                        else:
                            signleCloumnCount = signleCloumnCount + 1
                            xColumnsSQL = xColumnsSQL + columnSQL
                    if signleCloumnCount == 1:
                        sqlInfo = {}
                        sqlInfoArr.append(sqlInfo)

                    sqlInfo["InfoKeysSQL"] = infoKeysSQL
                    sqlInfo["YColumnsSQL"] = yColumnsSQL
                    sqlInfo["XColumnsSQL"] = xColumnsSQL
                    sqlInfo["WheresSQL"] = wheresSQL
                    sqlInfo["GroupKeysSQL"] = groupKeysSQL

                    if signleCloumnCount == makeMaxCloumnCount:
                        signleCloumnCount = 0
                        xColumnsSQL = ""

        dfArr = []
        dfIDArr = []
        for sqlInfo in sqlInfoArr :
            sql = makeSingleTagDataSQL(fvInfo,sqlInfo)
            print(sql)
            df = postgresCtrl.searchSQL(sql)
            dfIDArr.append(id(df))
            dfArr.append(df)
        return dfIDArr , dfArr

    # ==================================================  ExeSQLStrs ==================================================

    @classmethod
    def makeExeSQLStrsByDataBase(self,fvInfo,otherInfo):
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
        for sqlReplace in sqlReplaceArr :
            sqlStrs = sqlStrs.replace(sqlReplace[0],sqlReplace[1])
        sqlStrArr = sqlStrs.split(";")[:-1]
        for sqlStr in sqlStrArr:
            postgresCtrl.executeSQL(sqlStr)
        return {}


    # ==================================================  MakeTagText ==================================================

    @classmethod
    def makeTagTextByDataBase(self,fvInfo,otherInfo):
        from package.artificialintelligence.virtualentity.TagText import TagText
        tagText = TagText()
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database="scientificanalysis"
            , schema="public"
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