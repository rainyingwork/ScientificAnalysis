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
        elif functionVersionInfo['FunctionType'] == "DataConcat":
            otherInfo = self.ppDataConcat(functionVersionInfo)
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
    def ppDataConcat(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.makeDFByPPDataConcat(fvInfo, otherInfo)
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
                    , '[:DataType]' as datatype
                    , common_013 as index
                    , common_015 as jsonmessage
                FROM observationdata.analysisdata AA
                where 1 = 1
                   AND AA.product = '[:Product]'
                   AND AA.project='[:Project]'
                   AND AA.version='[:TextVersion]'
                   AND AA.dt = '[:DTStr]'
            """.replace("[:Product]", makeDataInfo['Product']) \
                .replace("[:Project]", makeDataInfo['Project']) \
                .replace("[:Version]", makeDataInfo['Version']) \
                .replace("[:TextVersion]", tagTextVersion) \
                .replace("[:DTStr]", makeDataInfo['DTStr']) \
                .replace("[:DataType]", makeDataInfo["DataType"])

            tempTextDF = postgresCtrl.searchSQL(tagTextSQL)
            if 'ColumnNumbers' in makeDataInfo.keys():
                stColumnNumbers = []
                for columnNumbers in makeDataInfo['ColumnNumbers']:
                    stColumnNumbers.append(str(columnNumbers))
                tempTextDF = tempTextDF[tempTextDF['index'].isin(stColumnNumbers)]
            tagTextDF = pandas.concat([tagTextDF, tempTextDF])
        tagTextDF.drop_duplicates()
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

        for mdInfo in fvInfo['MakeDataInfo']:
            for columnNumber in mdInfo['ColumnNumbers']:
                columnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(mdInfo["Product"], mdInfo["Project"],mdInfo['DTNameStr'], str(columnNumber),mdInfo['GFunc'], mdInfo["Version"]))
                for _, tagTextRow in tagTextDF.iterrows():
                    if (tagTextRow["product"] != mdInfo["Product"]) \
                        | (tagTextRow["project"] != mdInfo["Project"]) \
                        | (tagTextRow["dt"] != mdInfo["DTStr"]) \
                        | (tagTextRow["project"] != mdInfo["Project"]) \
                        | (tagTextRow["version"] != mdInfo["Version"]):
                        continue
                    if columnFullName not in preprossDF.columns:
                        preprossDF[columnFullName] = None
                    jsonMessage = json.loads(tagTextRow['jsonmessage'])
                    processingOrderArr = jsonMessage['DataPreProcess']['ProcessingOrder']
                    processingFunctions = jsonMessage['DataPreProcess']['ProcessingFunction']
                    for processingFunctionName in processingOrderArr:
                        processingFunction = processingFunctions[processingFunctionName]
                        if processingFunctionName == "fillna":
                            preprossDF[columnFullName] = preprossDF[columnFullName].fillna(processingFunction['value'])
                        elif processingFunctionName == "log":
                            preprossDF[columnFullName] = preprossDF[columnFullName].apply(lambda x: math.log(x, processingFunction['value']), axis=1)
        resultPreprossDFArr.append(preprossDF)
        return preprossDFArr

    @classmethod
    def makeDFByPPDataConcat(self, fvInfo, otherInfo):
        oriPreprossDFArr = fvInfo["ResultArr"]
        preprossDFArr = []
        preprossDF = pandas.DataFrame()
        for oriPreprossDF in oriPreprossDFArr :
            preprossDF = pandas.concat([preprossDF, oriPreprossDF],ignore_index=True)
        preprossDFArr.append(preprossDF)
        return preprossDFArr