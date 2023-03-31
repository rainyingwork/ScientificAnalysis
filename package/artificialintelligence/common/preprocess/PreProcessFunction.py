import os
import datetime
import math, pandas
import json
from dotenv import load_dotenv
from package.common.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class PreProcessFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["ResultArr"] :
                resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "PPTagText":
            otherInfo = self.ppTagText(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "ExeSQLStrs":
            otherInfo = self.ppExeSQLStrs(functionVersionInfo)
            resultDict["SQLStrs"] = ""
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def ppTagText(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.makeDFByPPTagText(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def ppExeSQLStrs(self, fvInfo):
        otherInfo = {}
        self.makeExeSQLStrsByDataBase(fvInfo, otherInfo)
        return otherInfo

    # ================================================= CommonFunction =================================================

    @classmethod
    def makeTagTextDF(self,fvInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        makeDataInfoArr = fvInfo['MakeDataInfo']

        tagTextDF = pandas.DataFrame()
        for makeDataInfo in makeDataInfoArr:
            tagTextVersion = makeDataInfo['Version'].split("_")[0] + '_' + makeDataInfo['Version'].split("_")[1] + '_0'
            tagTextSQL = """
                        SELECT
                            product , project , version as textversion , dt
                            , '[:Version]' as version
                            , [:DTDiff] as dtdiff
                            , '[:DataType]' as datatype
                            , common_013 as index
                            , common_015 as jsonmessage
                        FROM observationdata.analysisdata AA
                        where 1 = 1
                           AND AA.product = '[:Product]'
                           AND AA.project='[:Project]'
                           AND AA.version='[:TextVersion]'
                           AND AA.dt = to_char((date '[:MakeDataDateStr]' + integer '[:DTDiff]'),'yyyyMMdd')
                    """.replace("[:Product]", makeDataInfo['Product']) \
                .replace("[:Project]", makeDataInfo['Project']) \
                .replace("[:Version]", makeDataInfo['Version']) \
                .replace("[:TextVersion]", tagTextVersion) \
                .replace("[:MakeDataDateStr]", fvInfo["DataTime"]) \
                .replace("[:DTDiff]", str(makeDataInfo["DTDiff"])) \
                .replace("[:DataType]", makeDataInfo["DataType"])
            tempTextDF = postgresCtrl.searchSQL(tagTextSQL)
            if 'ColumnNumbers' in makeDataInfo.keys():
                stColumnNumbers = []
                for columnNumbers in makeDataInfo['ColumnNumbers']:
                    stColumnNumbers.append(str(columnNumbers))
                tempTextDF = tempTextDF[tempTextDF['index'].isin(stColumnNumbers)]
            tagTextDF = pandas.concat([tagTextDF, tempTextDF])
        return tagTextDF



    # ==================================================   PPTagText  ==================================================

    @classmethod
    def makeDFByPPTagText(self, fvInfo, otherInfo):

        preprossDFArr = fvInfo["ResultArr"]

        # -------------------------------------------------- tagTextDF--------------------------------------------------
        tagTextDF = self.makeTagTextDF(fvInfo)

        # --------------------------------------------------preprossDF--------------------------------------------------

        resultPreprossDFArr = []

        preprossDF = preprossDFArr[0]
        for _, tagTextRow in tagTextDF.iterrows():
            dtDiffStr = "p" + str(abs(tagTextRow["dtdiff"])) if tagTextRow["dtdiff"] >= 0 else "n" + str(
                abs(tagTextRow["dtdiff"]))
            tagTextColumnName = str.lower(
                "{}_{}_{}_{}_{}".format(tagTextRow["product"], tagTextRow["project"], dtDiffStr, tagTextRow["index"],
                                        tagTextRow["version"]))
            if tagTextColumnName not in preprossDF.columns:
                preprossDF[tagTextColumnName] = None
            jsonMessage = json.loads(tagTextRow['jsonmessage'])
            processingOrderArr = jsonMessage['DataPreProcess']['ProcessingOrder']
            processingFunctions = jsonMessage['DataPreProcess']['ProcessingFunction']
            for processingFunctionName in processingOrderArr:
                processingFunction = processingFunctions[processingFunctionName]
                if processingFunctionName == "fillna":
                    preprossDF[tagTextColumnName] = preprossDF[tagTextColumnName].fillna(processingFunction['value'])
                elif processingFunctionName == "log":
                    preprossDF[tagTextColumnName] = preprossDF[tagTextColumnName].apply(
                        lambda x: math.log(x, processingFunction['value']), axis=1)
        resultPreprossDFArr.append(preprossDF)
        return preprossDFArr