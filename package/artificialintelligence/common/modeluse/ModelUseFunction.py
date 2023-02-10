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
        databaseResultJson = self.getDatabaseResultJson(fvInfo)
        fvInfo["MakeDataKeys"] = databaseResultJson["MakeDataKeys"]
        fvInfo["MakeDataInfo"] = databaseResultJson["MakeDataInfo"]
        fvInfo["ModelParameter"] = databaseResultJson["ModelParameter"]
        fvInfo["ModelDesign"] = databaseResultJson["ModelDesign"]
        for modelDist in databaseResultJson["ModelDesign"]["ModelDists"] :
            if modelDist["ModelName"] != fvInfo["DatabaseModelName"] :
                continue
            fvInfo["ModelDesign"]["ModelDists"] = [modelDist]
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
                yMakeDataInfoArr.append(makeDataInfo) if makeDataInfo["DataType"] == "Y" else xMakeDataInfoArr.append(
                    makeDataInfo)
            for modeluseColumn in modeluseDF.columns:
                if modeluseColumn in makeDataKeys:
                    continue
                keyArr = modeluseColumn.split("_")
                product, project, version, dtdiffstr, columnNumber = keyArr[0], keyArr[1], keyArr[4] + '_' + keyArr[5] + '_' + keyArr[6], keyArr[2], int(keyArr[3])
                dtfiff = int(dtdiffstr[1:]) if dtdiffstr[0] == str.lower("P") else int(dtdiffstr[1:]) * -1
                for yInfo in yMakeDataInfoArr:
                    if product == str.lower(yInfo['Product']) and \
                            project == str.lower(yInfo['Project']) and \
                            version == str.lower(yInfo['Version']) and \
                            dtfiff == yInfo['DTDiff'] and \
                            (columnNumber in yInfo['ColumnNumbers'] or yInfo['ColumnNumbers'] == []):
                        yColumnNames.append(modeluseColumn)
                        df[modeluseColumn] = modeluseDF[modeluseColumn]
                for xInfo in xMakeDataInfoArr:
                    if product == str.lower(xInfo['Product']) and \
                            project == str.lower(xInfo['Project']) and \
                            version == str.lower(xInfo['Version']) and \
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
        resultInfo['MakeDataKeys'] = makeDataKeys
        resultInfo['MakeDataInfo'] = []
        for makeDataInfo in makeDataInfoArr :
            resultInfo['MakeDataInfo'].append(makeDataInfo) if makeDataInfo["DataType"] == "Y" else None

        for lcvrIndex, lcvrRow in lassoCVRusultDF.iterrows():
            keyArr = lcvrRow['columnname'].split("_")
            product, project, version, dtdiffstr, columnNumber = keyArr[0], keyArr[1], keyArr[4] + '_' + keyArr[5] + '_' + keyArr[6], keyArr[2], int(keyArr[3])
            columnValue = lcvrRow['columnvalue']
            dtfiff = int(dtdiffstr[1:]) if dtdiffstr[0] == "P" else int(dtdiffstr[1:]) * -1
            dt = (datetime.datetime.strptime(fvInfo["DataTime"], "%Y-%m-%d") + datetime.timedelta(days=dtfiff)).strftime("%Y%m%d")
            isInNewMakeDataInfo = False
            for makeDataInfo in resultInfo['MakeDataInfo'] :
                if product != str.lower(makeDataInfo['Product']) or \
                        project != str.lower(makeDataInfo['Project']) or \
                        version != str.lower(makeDataInfo['Version']) or \
                        dtfiff != makeDataInfo['DTDiff'] or \
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
                            dtfiff != makeDataInfo['DTDiff'] or \
                            'X' != makeDataInfo['DataType'] :
                        continue
                    tempMakeDataInfo = copy.deepcopy(makeDataInfo)
                    resultInfo["MakeDataInfo"].append(tempMakeDataInfo)
                    tempMakeDataInfo['DT'] = dt
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

        trainDF, testDF = train_test_split(df, test_size=0.2)
        setup(data=trainDF, target=yColumnNames[0], use_gpu=False)
        bestmodels = compare_models(sort='F1', n_select=10)

        resultDict = {}
        resultDict['MakeDataKeys'] = fvInfo['MakeDataKeys']
        resultDict['MakeDataInfo'] = fvInfo['MakeDataInfo']
        resultDict['ModelDesign'] = {}
        resultDict['ModelDesign']['ModelType'] = "AutoML"
        resultDict['ModelDesign']['ModelFunction'] = "UsePycaretDefult"
        resultDict['ModelDesign']['ModelDists'] = []
        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        for bestmodel in bestmodels:
            predictions = predict_model(bestmodel, data=testDF)
            tn, fp, fn, tp = confusion_matrix(predictions[yColumnNames[0]], predictions['Label']).ravel()
            modeldist = {}
            modeldist['ModelFullName'] = "{}.{}".format(bestmodel.__class__.__module__, bestmodel.__class__.__name__)
            modeldist['ModelPackage'] = bestmodel.__class__.__module__
            modeldist['ModelName'] = bestmodel.__class__.__name__
            modeldist['ModelFileName'] = modeldist['ModelName'] + ".pkl"
            modeldist['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
            modeldist['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir,modeldist['ModelFileName'])
            modeldist['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
            modeldist['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir, modeldist['ModelFileName'])
            modeldist['ModelResult'] = {}
            modeldist['ModelResult']['TN'] = int(tn)
            modeldist['ModelResult']['FP'] = int(fp)
            modeldist['ModelResult']['FN'] = int(fn)
            modeldist['ModelResult']['TP'] = int(tp)
            modeldist['ModelResult']['Accuracy'] = tp / (tp + fp)
            modeldist['ModelResult']['Precision'] = tp / (tp + fn)
            modeldist['ModelResult']['Recall'] = (tp + tn) / (tp + tn + fp + fn)
            modeldist['ModelResult']['F1Score'] = 2 * modeldist['ModelResult']['Recall'] * modeldist['ModelResult']['Precision'] / (modeldist['ModelResult']['Recall'] + modeldist['ModelResult']['Precision'])
            save_model(bestmodel, "{}/{}".format(modeldist['ModelStorageLocationPath'], modeldist['ModelName']))

            sshCtrl.execCommand("mkdir -p {}".format(modeldist['ModelStorageRemotePath']))
            sshCtrl.uploadFile(modeldist['ModelStorageLocation'],modeldist['ModelStorageRemote'])

            resultDict['ModelDesign']['ModelDists'].append(modeldist)
        shutil.rmtree("catboost_info")
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

        trainDF, testDF = train_test_split(df, test_size=0.2)
        setup(data=trainDF, target=yColumnNames[0], use_gpu=False)
        bestmodels = compare_models(sort='R2', n_select=10)

        resultDict = {}
        resultDict['MakeDataKeys'] = fvInfo['MakeDataKeys']
        resultDict['MakeDataInfo'] = fvInfo['MakeDataInfo']
        resultDict['ModelDesign'] = {}
        resultDict['ModelDesign']['ModelType'] = "AutoML"
        resultDict['ModelDesign']['ModelFunction'] = "UsePycaretDefult"
        resultDict['ModelDesign']['ModelDists'] = []
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
            modeldist = {}
            modeldist['ModelFullName'] = "{}.{}".format(bestmodel.__class__.__module__, bestmodel.__class__.__name__)
            modeldist['ModelPackage'] = bestmodel.__class__.__module__
            modeldist['ModelName'] = bestmodel.__class__.__name__
            modeldist['ModelFileName'] = modeldist['ModelName'] + ".pkl"
            modeldist['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
            modeldist['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldist['ModelFileName'])
            modeldist['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
            modeldist['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldist['ModelFileName'])
            modeldist['ModelResult'] = {}
            modeldist['ModelResult']['MAE'] = float(MAE)
            modeldist['ModelResult']['MSE'] = float(MSE)
            modeldist['ModelResult']['RMSE'] = float(RMSE)
            modeldist['ModelResult']['R2'] = float(R2)
            modeldist['ModelResult']['EVS'] = float(EVS)

            save_model(bestmodel, "{}/{}".format(modeldist['ModelStorageLocationPath'], modeldist['ModelName']))

            sshCtrl.execCommand("mkdir -p {}".format(modeldist['ModelStorageRemotePath']))
            sshCtrl.uploadFile(modeldist['ModelStorageLocation'], modeldist['ModelStorageRemote'])

            resultDict['ModelDesign']['ModelDists'].append(modeldist)
        shutil.rmtree("catboost_info")
        return resultDict

    @classmethod
    def makeAutoMLByUsePycaretModelClassification(self, fvInfo, otherInfo):
        from sklearn.metrics import confusion_matrix
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
        modeldist = fvInfo["ModelDesign"]['ModelDists'][0]

        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        modeldist['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
        modeldist['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldist['ModelFileName'])

        sshCtrl.downloadFile(modeldist['ModelStorageRemote'], modeldist['ModelStorageLocation'])
        bestmodel = load_model(modeldist['ModelStorageLocation'].replace(".pkl", ""))

        predictions = predict_model(bestmodel, data=df)

        oriDF = fvInfo["ResultArr"][0]
        resultDF = oriDF[commonColumnNames]
        resultDF[self.getDoubleColumnArr()[0]] = predictions[yColumnNames[0]]
        resultDF[self.getDoubleColumnArr()[1]] = predictions['Label']

        tn, fp, fn, tp = confusion_matrix(predictions[yColumnNames[0]], predictions['Label']).ravel()

        modeldist['ModelResult']['TN'] = int(tn)
        modeldist['ModelResult']['FP'] = int(fp)
        modeldist['ModelResult']['FN'] = int(fn)
        modeldist['ModelResult']['TP'] = int(tp)
        modeldist['ModelResult']['Accuracy'] = tp / (tp + fp)
        modeldist['ModelResult']['Precision'] = tp / (tp + fn)
        modeldist['ModelResult']['Recall'] = (tp + tn) / (tp + tn + fp + fn)
        modeldist['ModelResult']['F1Score'] = 2 * modeldist['ModelResult']['Recall'] * modeldist['ModelResult']['Precision'] / (modeldist['ModelResult']['Recall'] + modeldist['ModelResult']['Precision'])

        modeldist['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
        modeldist['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldist['ModelFileName'])

        sshCtrl.execCommand("mkdir -p {}".format(modeldist['ModelStorageRemotePath']))
        sshCtrl.uploadFile(modeldist['ModelStorageLocation'], modeldist['ModelStorageRemote'])
        resultDict = copy.deepcopy(fvInfo)
        resultDict["ResultArr"] = [resultDF]
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

        modeldist = fvInfo["ModelDesign"]['ModelDists'][0]

        product, project, opsVersion, opsRecordId, executeFunction = fvInfo["Product"],fvInfo["Project"],fvInfo["OPSVersion"],str(fvInfo["OPSRecordId"]),fvInfo["Version"]
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, opsRecordId,executeFunction)
        exeFunctionRDir = "{}/{}/{}/{}/{}".format(product, project,opsVersion,opsRecordId,executeFunction)
        os.makedirs(exeFunctionLDir) if not os.path.isdir(exeFunctionLDir) else None
        modeldist['ModelStorageLocationPath'] = "{}".format(exeFunctionLDir)
        modeldist['ModelStorageLocation'] = "{}/{}".format(exeFunctionLDir, modeldist['ModelFileName'])

        sshCtrl.downloadFile(modeldist['ModelStorageRemote'], modeldist['ModelStorageLocation'])
        bestmodel = load_model(modeldist['ModelStorageLocation'].replace(".pkl",""))

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
        modeldist['ModelResult']['MAE'] = float(MAE)
        modeldist['ModelResult']['MSE'] = float(MSE)
        modeldist['ModelResult']['RMSE'] = float(RMSE)
        modeldist['ModelResult']['R2'] = float(R2)
        modeldist['ModelResult']['EVS'] = float(EVS)

        modeldist['ModelStorageRemotePath'] = "/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir)
        modeldist['ModelStorageRemote'] = "/{}/{}/{}".format(os.getenv("STORAGE_RECORDSAVEPATH"),exeFunctionRDir,modeldist['ModelFileName'])

        sshCtrl.execCommand("mkdir -p {}".format(modeldist['ModelStorageRemotePath']))
        sshCtrl.uploadFile(modeldist['ModelStorageLocation'], modeldist['ModelStorageRemote'])
        resultDict = copy.deepcopy(fvInfo)
        resultDict["ResultArr"] = [resultDF]
        return resultDict

