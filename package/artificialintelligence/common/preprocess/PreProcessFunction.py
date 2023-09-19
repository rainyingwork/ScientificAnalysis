import os
import datetime
import math, pandas
import json
from dotenv import load_dotenv
from package.common.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction
from package.artificialintelligence.common.preprocess.PreProcessTool import PreProcessTool

class PreProcessFunction(PreProcessTool,CommonFunction):

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

    # ==================================================   PPTagText  ==================================================

    @classmethod
    def makeDFByPPTagText(self, fvInfo, otherInfo):

        preprossDFArr = fvInfo["ResultArr"]

        # --------------------------------------------------preprossDF--------------------------------------------------

        preprossDF = preprossDFArr[0]

        for mdInfo in fvInfo['MakeDataInfo']:
            for columnNumber in mdInfo['ColumnNumbers']:
                columnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(mdInfo["Product"], mdInfo["Project"],mdInfo['DTNameStr'], str(columnNumber),mdInfo['GFunc'], mdInfo["Version"]))
                jsonMessage = mdInfo['ColumnInfo'][str(columnNumber)]['messageinfo']
                processingOrderArr = jsonMessage['DataPreProcess']['ProcessingOrder']
                processingFunctions = jsonMessage['DataPreProcess']['ProcessingFunction']
                for processingFunctionName in processingOrderArr:
                    processingFunction = processingFunctions[processingFunctionName]
                    if processingFunctionName == "fillna":
                        preprossDF[columnFullName] = preprossDF[columnFullName].fillna(processingFunction['value']) if columnFullName in preprossDF.columns else processingFunction['value']
                    elif processingFunctionName == "log":
                        preprossDF[columnFullName] = preprossDF[columnFullName].apply(lambda x: (math.log(x, processingFunction['value']) if x > 0 else 0 ))
                    elif processingFunctionName == "rank":
                        preprossDF[columnFullName] = preprossDF[columnFullName].rank()
                    elif processingFunctionName == "normbymaxmin":
                        columnMax = preprossDF[columnFullName].max()
                        columnMin = preprossDF[columnFullName].min()
                        preprossDF[columnFullName] = preprossDF[columnFullName].apply(lambda x: (x - columnMin) / (columnMax - columnMax)) if (columnMax - columnMax) == 0 else 0
                    elif processingFunctionName == "normbyzscore":
                        columnMean = preprossDF[columnFullName].mean()
                        columnStd = preprossDF[columnFullName].std()
                        preprossDF[columnFullName] = preprossDF[columnFullName].apply(lambda x: (x - columnMean) / columnStd) if columnStd == 0 else 0
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
