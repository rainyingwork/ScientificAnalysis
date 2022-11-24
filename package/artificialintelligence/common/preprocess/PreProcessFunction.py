import os
import time , datetime
import pprint
import math, pandas
import json
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class PreProcessFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "PPTagText":
            otherInfo = self.ppTagText(functionVersionInfo)
            resultDict['ResultIDArr'] , globalObjectDict['ResultArr'] = otherInfo['DFIDArr'] , otherInfo["DFArr"]
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def ppTagText(self, fvInfo):
        otherInfo = {}
        otherInfo["DFIDArr"], otherInfo["DFArr"] = self.makeDFByPPTagText(fvInfo, otherInfo)
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
            , database="scientificanalysis"
            , schema="public"
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
            tagTextDF = pandas.concat([tagTextDF, postgresCtrl.searchSQL(tagTextSQL)])
        return tagTextDF



    # ==================================================   PPTagText  ==================================================

    @classmethod
    def makeDFByPPTagText(self, fvInfo, otherInfo):

        preprossDFIDArr = fvInfo["ResultIDArr"]
        makeDataKeys = fvInfo['MakeDataKeys']
        makeDataInfoArr = fvInfo['MakeDataInfo']

        # -------------------------------------------------- tagTextDF--------------------------------------------------

        tagTextDF = self.makeTagTextDF(fvInfo)

        # --------------------------------------------------preprossDF--------------------------------------------------

        preprossDFArr = []
        for preprossDFID in preprossDFIDArr:
            preprossDFArr.append(GainObjectCtrl.getObjectsById(preprossDFID))

        preprossResultDFIDArr = []
        preprossResultDFArr = []
        for preprossDF in preprossDFArr :
            filter = preprossDF[preprossDF.columns[0]] != '~!@#$%^&*()_++_)(*&^%$#@!~'
            for columnName in preprossDF.columns:
                if columnName in makeDataKeys:
                    continue
                keyArr = columnName.split("_")
                product, project, version, dtdiiffstr, columnNumber = keyArr[0], keyArr[1], keyArr[4] + '_' + keyArr[5] + '_' + keyArr[6], keyArr[2], int(keyArr[3])
                dtfiff = int(dtdiiffstr[1:]) if dtdiiffstr[0] == str.lower("P") else int(dtdiiffstr[1:]) * -1
                dt = (datetime.datetime.strptime(fvInfo['DataTime'], "%Y-%m-%d") + datetime.timedelta(days=dtfiff)).strftime("%Y%m%d")
                for _, tagTextRow in tagTextDF.iterrows():
                    if product != str.lower(tagTextRow['product']) or \
                            project != str.lower(tagTextRow['project']) or \
                            version != str.lower(tagTextRow['version']) or \
                            columnNumber != int(tagTextRow['index']) or \
                            dtfiff != tagTextRow['dtdiff']:
                        continue
                    if tagTextRow['datatype'] == 'X':
                        filter = filter & preprossDF[columnName].isna()
                    jsonMessage = json.loads(tagTextRow['jsonmessage'])
                    processingOrderArr = jsonMessage['DataPreProcess']['ProcessingOrder']
                    processingFunctions = jsonMessage['DataPreProcess']['ProcessingFunction']
                    for processingFunctionName in processingOrderArr:
                        processingFunction = processingFunctions[processingFunctionName]
                        if processingFunctionName == "fillna":
                            preprossDF[columnName] = preprossDF[columnName].fillna(processingFunction['value'])
                        elif processingFunctionName == "log":
                            preprossDF[columnName] = preprossDF[columnName].apply(lambda x: math.log(x, processingFunction['value']), axis=1)
            preprossDF = preprossDF[~filter]
            preprossResultDFIDArr.append(id(preprossDF))
            preprossResultDFArr.append(preprossDF)

        return preprossResultDFIDArr , preprossResultDFArr
