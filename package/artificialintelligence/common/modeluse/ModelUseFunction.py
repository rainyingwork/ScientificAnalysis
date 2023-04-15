import os , copy
import datetime
import numpy ,pandas
from dotenv import load_dotenv
from package.common.common.osbasic.SSHCtrl import SSHCtrl
from package.artificialintelligence.common.common.CommonFunction import CommonFunction

class ModelUseFunction(CommonFunction):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        for key in functionVersionInfo.keys():
            if key not in ["ResultArr"] :
                resultDict[key] = functionVersionInfo[key]
        if functionVersionInfo['FunctionType'] == "TagFilter":
            if functionVersionInfo['ModelFunction'] == "Lasso":
                otherInfo = self.muTagFilterByLasso(functionVersionInfo)
                resultDict = otherInfo['ResultInfo']
        elif functionVersionInfo['FunctionType'] == "AutoML":
            if functionVersionInfo['ModelFunction'] == "TrainPycaretDefult":
                otherInfo = self.muAutoMLByTrainPycaretDefult(functionVersionInfo)
                resultDict = otherInfo['ResultInfo']
            elif functionVersionInfo['ModelFunction'] == "UsePycaretModelByDatabaseRusult":
                otherInfo = self.muAutoMLByUsePycaretModelByDatabaseRusult(functionVersionInfo)
                resultDict = otherInfo['ResultInfo']
                globalObjectDict["ResultArr"] = resultDict["ResultArr"] ; resultDict.pop("ResultArr")
        elif functionVersionInfo['FunctionType'] == "ExeSQLStrs":
            otherInfo = self.muExeSQLStrs(functionVersionInfo)
            resultDict["SQLStrs"] = ""
        resultDict['Result'] = "OK"
        return resultDict, globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def muTagFilterByLasso(self, fvInfo):
        otherInfo = {}
        otherInfo["ResultInfo"] = self.makeTagFilterByLasso(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def muAutoMLByTrainPycaretDefult(self, fvInfo):
        otherInfo = {}
        if fvInfo["ModelParameter"]["TaskType"] == "Classification" :
            otherInfo["ResultInfo"] = self.makeAutoMLByTrainPycaretDefultClassification(fvInfo, otherInfo)
        elif fvInfo["ModelParameter"]["TaskType"] == "Regression" :
            otherInfo["ResultInfo"] = self.makeAutoMLByTrainPycaretDefultRegression(fvInfo, otherInfo)
        otherInfo["ResultInfo"]['ModelParameter'] = fvInfo['ModelParameter']
        return otherInfo

    @classmethod
    def muAutoMLByUsePycaretModelByDatabaseRusult(self, fvInfo):
        otherInfo = {}
        # databaseResultJson = self.getDatabaseResultJson(fvInfo)
        # fvInfo["MakeDataKeys"] = databaseResultJson["MakeDataKeys"]
        # fvInfo["MakeDataInfo"] = databaseResultJson["MakeDataInfo"]
        # fvInfo["ModelParameter"] = databaseResultJson["ModelParameter"]
        # fvInfo["ModelDesign"] = databaseResultJson["ModelDesign"]
        # for modelDict in databaseResultJson["ModelDesign"]["ModelDicts"] :
        #     if modelDict["ModelName"] != fvInfo["DatabaseModelName"] :
        #         continue
        #     fvInfo["ModelDesign"]["ModelDicts"] = [modelDict]
        if fvInfo["ModelParameter"]["TaskType"] == "Classification":
            otherInfo["ResultInfo"] = self.makeAutoMLByUsePycaretModelClassification(fvInfo, otherInfo)
        elif fvInfo["ModelParameter"]["TaskType"] == "Regression":
            otherInfo["ResultInfo"] = self.makeAutoMLByUsePycaretModelRegression(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def muExeSQLStrs(self, fvInfo):
        otherInfo = {}
        self.makeExeSQLStrsByDataBase(fvInfo, otherInfo)
        return otherInfo

    # ================================================= CommonFunction =================================================

    @classmethod
    def makeXYDataInfoAndColumnNames (self , fvInfo, otherInfo) :

        modeluseDFArr = fvInfo["ResultArr"]

        makeDataKeys = fvInfo['MakeDataKeys']
        makeDataInfoArr = fvInfo['MakeDataInfo']
        df = pandas.DataFrame()
        yMakeDataInfoArr, xMakeDataInfoArr = [], []
        commonColumnNames, yColumnNames, xColumnNames = makeDataKeys, [], []
        for modeluseDF in modeluseDFArr:
            for makeDataInfo in makeDataInfoArr:
                yMakeDataInfoArr.append(makeDataInfo) if makeDataInfo["DataType"] == "Y" else xMakeDataInfoArr.append(makeDataInfo)
            for modeluseColumn in modeluseDF.columns:
                if modeluseColumn in makeDataKeys:
                    continue
                keyArr = modeluseColumn.split("_")
                product, project, version, dtdiffstr, gfunc , columnNumber = keyArr[0], keyArr[1], keyArr[5] + '_' + keyArr[6] + '_' + keyArr[7], keyArr[2], keyArr[4], int(keyArr[3])
                dtfiff = int(dtdiffstr[1:]) if dtdiffstr[0] == str.lower("P") else int(dtdiffstr[1:]) * -1
                for yInfo in yMakeDataInfoArr:
                    if product == str.lower(yInfo['Product']) and \
                            project == str.lower(yInfo['Project']) and \
                            version == str.lower(yInfo['Version']) and \
                            gfunc == str.lower(yInfo['GFunc']) and \
                            dtfiff == yInfo['DTDiff'] and \
                            (columnNumber in yInfo['ColumnNumbers'] or yInfo['ColumnNumbers'] == []):
                        yColumnNames.append(modeluseColumn)
                        df[modeluseColumn] = modeluseDF[modeluseColumn]
                for xInfo in xMakeDataInfoArr:
                    if product == str.lower(xInfo['Product']) and \
                            project == str.lower(xInfo['Project']) and \
                            version == str.lower(xInfo['Version']) and \
                            gfunc == str.lower(xInfo['GFunc']) and \
                            dtfiff == xInfo['DTDiff'] and \
                            (columnNumber in xInfo['ColumnNumbers'] or xInfo['ColumnNumbers'] == []):
                        xColumnNames.append(modeluseColumn)
                        df[modeluseColumn] = modeluseDF[modeluseColumn]
                yColumnNames = list(set(yColumnNames))
                xColumnNames = list(set(xColumnNames))
        return df , yMakeDataInfoArr, xMakeDataInfoArr , commonColumnNames, yColumnNames, xColumnNames

    # ==================================================     Lasso    ==================================================

    @classmethod
    def makeTagFilterByLasso(self,fvInfo, otherInfo):
        from sklearn.linear_model import LassoCV
        makeDataKeys = fvInfo['MakeDataKeys']
        makeDataInfoArr = fvInfo['MakeDataInfo']

        modelParameter = fvInfo['ModelParameter']
        modelParameterTopK = modelParameter['TopK'] if 'TopK' in modelParameter.keys() else None
        modelParameterFilter = modelParameter['Filter'] if 'Filter' in modelParameter.keys() else None

        # -------------------------------------------------- XYColumn --------------------------------------------------

        df , yMakeDataInfoArr, xMakeDataInfoArr , commonColumnNames, yColumnNames, xColumnNames = \
            self.makeXYDataInfoAndColumnNames(fvInfo, otherInfo)

        # -------------------------------------------------- MakeLasso--------------------------------------------------
        trainY = df[yColumnNames]
        trainX = df[xColumnNames]

        lrcv = LassoCV(alphas=numpy.linspace(0.005, 1, 100), cv=3)
        lrcv.fit(trainX, trainY)
        corf = pandas.Series(lrcv.coef_, index=trainX.columns)
        sortCorf = abs(corf).sort_values(ascending=False)
        if modelParameterTopK != None:
            sortCorf = sortCorf[0:modelParameterTopK]
        if modelParameterFilter != None:
            sortCorf = sortCorf[sortCorf >= modelParameterFilter]
        lassoCVRusultDF = pandas.DataFrame(sortCorf).reset_index()
        lassoCVRusultDF.columns = ["columnname", "columnvalue"]

        # --------------------------------------------------MakeResult--------------------------------------------------

        resultInfo = {}
        resultInfo['FunctionItemType'] = fvInfo['FunctionItemType']
        resultInfo['MakeDataKeys'] = fvInfo['MakeDataKeys']
        resultInfo['MakeDataInfo'] = []
        for makeDataInfo in makeDataInfoArr :
            resultInfo['MakeDataInfo'].append(makeDataInfo) if makeDataInfo["DataType"] == "Y" else None
        for lcvrIndex, lcvrRow in lassoCVRusultDF.iterrows():
            keyArr = lcvrRow['columnname'].split("_")
            product, project, version, dtdiffstr, gfunc , columnNumber = keyArr[0], keyArr[1], keyArr[5] + '_' + keyArr[6] + '_' + keyArr[7], keyArr[2], keyArr[4], int(keyArr[3])
            columnValue = lcvrRow['columnvalue']
            dtfiff = int(dtdiffstr[1:]) if dtdiffstr[0] == "p" else int(dtdiffstr[1:]) * -1
            isInNewMakeDataInfo = False
            for makeDataInfo in resultInfo['MakeDataInfo'] :
                if product != str.lower(makeDataInfo['Product']) or \
                        project != str.lower(makeDataInfo['Project']) or \
                        version != str.lower(makeDataInfo['Version']) or \
                        gfunc != str.lower(makeDataInfo['GFunc']) or \
                        dtfiff != makeDataInfo['DTDiff'] or \
                        columnNumber not in makeDataInfo['ColumnNumbers'] or \
                        'X' != makeDataInfo['DataType'] :
                    continue
                if 'ColumnNumbers' not in makeDataInfo.keys():
                    makeDataInfo['ColumnNumbers'] = []
                    makeDataInfo['ColumnValues'] = []
                makeDataInfo['ColumnNumbers'].append(columnNumber)
                makeDataInfo['ColumnValues'].append(columnValue)
                isInNewMakeDataInfo = True
            if isInNewMakeDataInfo == False:
                for makeDataInfo in makeDataInfoArr :
                    if product != str.lower(makeDataInfo['Product']) or \
                            project != str.lower(makeDataInfo['Project']) or \
                            version != str.lower(makeDataInfo['Version']) or \
                            gfunc != str.lower(makeDataInfo['GFunc']) or \
                            dtfiff != makeDataInfo['DTDiff'] or \
                            columnNumber not in makeDataInfo['ColumnNumbers'] or \
                            'X' != makeDataInfo['DataType'] :
                        continue
                    tempMakeDataInfo = copy.deepcopy(makeDataInfo)
                    resultInfo["MakeDataInfo"].append(tempMakeDataInfo)
                    tempMakeDataInfo['ColumnNumbers'] = []
                    tempMakeDataInfo['ColumnValues'] = []
                    tempMakeDataInfo['ColumnNumbers'].append(columnNumber)
                    tempMakeDataInfo['ColumnValues'].append(columnValue)
        return resultInfo


    # =================================================UsePycaretDefult=================================================

    @classmethod
    def makeAutoMLByTrainPycaretDefultClassification(self, fvInfo, otherInfo):
        import os , shutil
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from pycaret.classification import setup
        from pycaret.classification import compare_models
        from pycaret.classification import predict_model
        from pycaret.classification import save_model

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

        # -------------------------------------------------- XYColumn --------------------------------------------------

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = \
            self.makeXYDataInfoAndColumnNames(fvInfo, otherInfo)

        # -------------------------------------------------- MakeModel--------------------------------------------------
        includeModel = fvInfo["ModelParameter"]["IncludeModel"] if "IncludeModel" in fvInfo["ModelParameter"].keys() else None
        excludeModel = fvInfo["ModelParameter"]["ExcludeModel"] if "ExcludeModel" in fvInfo["ModelParameter"].keys() and includeModel ==None else None
        topModelCount = fvInfo["ModelParameter"]["TopModelCount"] if "TopModelCount" in fvInfo["ModelParameter"].keys() else 10
        trainDF, testDF = train_test_split(df, test_size=0.2)
        setup(data=trainDF, target=yColumnNames[0], use_gpu=False)
        bestmodels = compare_models(sort='F1', n_select=topModelCount ,include=includeModel, exclude=excludeModel)

        resultDict = {}
        resultDict['MakeDataKeys'] = fvInfo['MakeDataKeys']
        resultDict['MakeDataInfo'] = fvInfo['MakeDataInfo']
        resultDict['ModelDesign'] = {}
        resultDict['ModelDesign']['ModelType'] = "AutoML"
        resultDict['ModelDesign']['ModelFunction'] = "UsePycaretDefult"
        resultDict['ModelDesign']['ModelDicts'] = []
        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        for bestmodel in bestmodels:
            predictions = predict_model(bestmodel, data=testDF)
            tp = int(((predictions[yColumnNames[0]] != 0) * (predictions['Label'] != 0)).sum())
            fp = int(((predictions[yColumnNames[0]] != 0) * (predictions['Label'] == 0)).sum())
            tn = int(((predictions[yColumnNames[0]] == 0) * (predictions['Label'] == 0)).sum())
            fn = int(((predictions[yColumnNames[0]] == 0) * (predictions['Label'] != 0)).sum())
            modeldict = {}
            modeldict['ModelFullName'] = "{}.{}".format(bestmodel.__class__.__module__, bestmodel.__class__.__name__)
            modeldict['ModelPackage'] = bestmodel.__class__.__module__
            modeldict['ModelName'] = bestmodel.__class__.__name__
            modeldict['ModelFileName'] = modeldict['ModelName'] + ".pkl"
            modeldict['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
            modeldict['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir,modeldict['ModelFileName'])
            modeldict['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
            modeldict['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir, modeldict['ModelFileName'])
            modeldict['ModelResult'] = {}
            modeldict['ModelResult']['TN'] = int(tn)
            modeldict['ModelResult']['FP'] = int(fp)
            modeldict['ModelResult']['FN'] = int(fn)
            modeldict['ModelResult']['TP'] = int(tp)
            modeldict['ModelResult']['Accuracy'] = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) != 0 else 0
            modeldict['ModelResult']['Precision'] = (tp / (tp + fp)) if (tp + fp) != 0 else 0
            modeldict['ModelResult']['Recall'] = (tp / (tp + fn)) if (tp + fn) != 0 else 0
            modeldict['ModelResult']['F1Score'] = ((2 * modeldict['ModelResult']['Precision'] * modeldict['ModelResult']['Recall']) /(modeldict['ModelResult']['Precision'] + modeldict['ModelResult']['Recall']) )if (modeldict['ModelResult']['Precision'] + modeldict['ModelResult']['Recall']) != 0 else 0
            save_model(bestmodel, "{}/{}".format(modeldict['ModelStorageLocationPath'], modeldict['ModelName']))
            sshCtrl.execCommand("mkdir -p {}".format(modeldict['ModelStorageRemotePath']))
            sshCtrl.uploadFile(modeldict['ModelStorageLocation'],modeldict['ModelStorageRemote'])

            resultDict['ModelDesign']['ModelDicts'].append(modeldict)
        del sshCtrl
        return resultDict

    @classmethod
    def makeAutoMLByTrainPycaretDefultRegression(self, fvInfo, otherInfo):
        import os , shutil
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        from pycaret.regression import setup
        from pycaret.regression import compare_models
        from pycaret.regression import predict_model
        from pycaret.regression import save_model

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

        # -------------------------------------------------- XYColumn --------------------------------------------------

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = \
            self.makeXYDataInfoAndColumnNames(fvInfo, otherInfo)

        # -------------------------------------------------- MakeModel--------------------------------------------------
        includeModel = fvInfo["ModelParameter"]["IncludeModel"] if "IncludeModel" in fvInfo["ModelParameter"].keys() else None
        excludeModel = fvInfo["ModelParameter"]["ExcludeModel"] if "ExcludeModel" in fvInfo["ModelParameter"].keys() and includeModel ==None else None
        topModelCount = fvInfo["ModelParameter"]["TopModelCount"] if "TopModelCount" in fvInfo["ModelParameter"].keys() else 10

        trainDF, testDF = train_test_split(df, test_size=0.2)
        setup(data=trainDF, target=yColumnNames[0], use_gpu=False)
        bestmodels = compare_models(sort='R2', n_select=topModelCount ,include=includeModel, exclude=excludeModel)

        resultDict = {}
        resultDict['MakeDataKeys'] = fvInfo['MakeDataKeys']
        resultDict['MakeDataInfo'] = fvInfo['MakeDataInfo']
        resultDict['ModelDesign'] = {}
        resultDict['ModelDesign']['ModelType'] = "AutoML"
        resultDict['ModelDesign']['ModelFunction'] = "UsePycaretDefult"
        resultDict['ModelDesign']['ModelDicts'] = []
        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        for bestmodel in bestmodels:
            predictions = predict_model(bestmodel, data=testDF)
            MAE = mean_absolute_error(predictions[yColumnNames[0]], predictions['Label'])
            MSE = mean_squared_error(predictions[yColumnNames[0]], predictions['Label'])
            RMSE = mean_squared_error(predictions[yColumnNames[0]], predictions['Label'], squared=False)
            R2 = r2_score(predictions[yColumnNames[0]], predictions['Label'])
            EVS = explained_variance_score(predictions[yColumnNames[0]], predictions['Label'])
            modeldict = {}
            modeldict['ModelFullName'] = "{}.{}".format(bestmodel.__class__.__module__, bestmodel.__class__.__name__)
            modeldict['ModelPackage'] = bestmodel.__class__.__module__
            modeldict['ModelName'] = bestmodel.__class__.__name__
            modeldict['ModelFileName'] = modeldict['ModelName'] + ".pkl"
            modeldict['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
            modeldict['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldict['ModelFileName'])
            modeldict['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
            modeldict['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldict['ModelFileName'])
            modeldict['ModelResult'] = {}
            modeldict['ModelResult']['MAE'] = float(MAE)
            modeldict['ModelResult']['MSE'] = float(MSE)
            modeldict['ModelResult']['RMSE'] = float(RMSE)
            modeldict['ModelResult']['R2'] = float(R2)
            modeldict['ModelResult']['EVS'] = float(EVS)

            save_model(bestmodel, "{}/{}".format(modeldict['ModelStorageLocationPath'], modeldict['ModelName']))

            sshCtrl.execCommand("mkdir -p {}".format(modeldict['ModelStorageRemotePath']))
            sshCtrl.uploadFile(modeldict['ModelStorageLocation'], modeldict['ModelStorageRemote'])

            resultDict['ModelDesign']['ModelDicts'].append(modeldict)

        del sshCtrl
        return resultDict

    @classmethod
    def makeAutoMLByUsePycaretModelClassification(self, fvInfo, otherInfo):
        from sklearn.metrics import confusion_matrix
        from pycaret.classification import predict_model
        from pycaret.classification import load_model
        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

        # -------------------------------------------------- XYColumn --------------------------------------------------

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = \
            self.makeXYDataInfoAndColumnNames(fvInfo, otherInfo)

        # -------------------------------------------------- MakeModel--------------------------------------------------
        modeldict = fvInfo["ModelDesign"]['ModelDicts'][0]
        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        modeldict['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
        modeldict['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldict['ModelFileName'])

        sshCtrl.downloadFile(modeldict['ModelStorageRemote'], modeldict['ModelStorageLocation'])
        bestmodel = load_model(modeldict['ModelStorageLocation'].replace(".pkl", ""))

        predictions = predict_model(bestmodel, data=df)

        oriDF = fvInfo["ResultArr"][0]
        resultDF = oriDF[commonColumnNames]
        resultDF[self.getDoubleColumnArr()[0]] = predictions[yColumnNames[0]]
        resultDF[self.getDoubleColumnArr()[1]] = predictions['Label']

        tp = int(((predictions[yColumnNames[0]] != 0) * (predictions['Label'] != 0)).sum())
        fp = int(((predictions[yColumnNames[0]] != 0) * (predictions['Label'] == 0)).sum())
        tn = int(((predictions[yColumnNames[0]] == 0) * (predictions['Label'] == 0)).sum())
        fn = int(((predictions[yColumnNames[0]] == 0) * (predictions['Label'] != 0)).sum())

        modeldict['ModelResult']['TN'] = int(tn)
        modeldict['ModelResult']['FP'] = int(fp)
        modeldict['ModelResult']['FN'] = int(fn)
        modeldict['ModelResult']['TP'] = int(tp)
        modeldict['ModelResult']['Accuracy'] = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) != 0 else 0
        modeldict['ModelResult']['Precision'] = (tp / (tp + fp)) if (tp + fp) != 0 else 0
        modeldict['ModelResult']['Recall'] = (tp / (tp + fn)) if (tp + fn) != 0 else 0
        modeldict['ModelResult']['F1Score'] = ((2 * modeldict['ModelResult']['Precision'] * modeldict['ModelResult']['Recall']) /(modeldict['ModelResult']['Precision'] + modeldict['ModelResult']['Recall']) )if (modeldict['ModelResult']['Precision'] + modeldict['ModelResult']['Recall']) != 0 else 0

        modeldict['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
        modeldict['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldict['ModelFileName'])

        sshCtrl.execCommand("mkdir -p {}".format(modeldict['ModelStorageRemotePath']))
        sshCtrl.uploadFile(modeldict['ModelStorageLocation'], modeldict['ModelStorageRemote'])
        resultDict = copy.deepcopy(fvInfo)
        resultDict["ResultArr"] = [resultDF]
        resultDict["ModelDict"] = modeldict
        del sshCtrl
        return resultDict

    @classmethod
    def makeAutoMLByUsePycaretModelRegression(self, fvInfo, otherInfo):
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        from pycaret.regression import predict_model
        from pycaret.regression import load_model

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

        # -------------------------------------------------- XYColumn --------------------------------------------------

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = \
            self.makeXYDataInfoAndColumnNames(fvInfo, otherInfo)

        # -------------------------------------------------- MakeModel--------------------------------------------------

        modeldict = fvInfo["ModelDesign"]['ModelDicts'][0]

        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        modeldict['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
        modeldict['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldict['ModelFileName'])

        sshCtrl.downloadFile(modeldict['ModelStorageRemote'], modeldict['ModelStorageLocation'])
        bestmodel = load_model(modeldict['ModelStorageLocation'].replace(".pkl",""))

        predictions = predict_model(bestmodel, data=df)

        oriDF = fvInfo["ResultArr"][0]
        resultDF = oriDF[commonColumnNames]
        resultDF[self.getDoubleColumnArr()[0]] = predictions[yColumnNames[0]]
        resultDF[self.getDoubleColumnArr()[1]] = predictions['Label']

        MAE = mean_absolute_error(predictions[yColumnNames[0]], predictions['Label'])
        MSE = mean_squared_error(predictions[yColumnNames[0]], predictions['Label'])
        RMSE = mean_squared_error(predictions[yColumnNames[0]], predictions['Label'], squared=False)
        R2 = r2_score(predictions[yColumnNames[0]], predictions['Label'])
        EVS = explained_variance_score(predictions[yColumnNames[0]], predictions['Label'])
        modeldict['ModelResult']['MAE'] = float(MAE)
        modeldict['ModelResult']['MSE'] = float(MSE)
        modeldict['ModelResult']['RMSE'] = float(RMSE)
        modeldict['ModelResult']['R2'] = float(R2)
        modeldict['ModelResult']['EVS'] = float(EVS)

        modeldict['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
        modeldict['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldict['ModelFileName'])

        sshCtrl.execCommand("mkdir -p {}".format(modeldict['ModelStorageRemotePath']))
        sshCtrl.uploadFile(modeldict['ModelStorageLocation'], modeldict['ModelStorageRemote'])
        resultDict = copy.deepcopy(fvInfo)
        resultDict["ResultArr"] = [resultDF]
        del sshCtrl
        return resultDict

