
class ModelUse() :

    @classmethod
    def M0_1_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_1_X"])
        if "ModelFullName" not in functionVersionInfo.keys() :
            return {}, {}

        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_1_X"
        functionVersionInfo["FunctionType"] = "AutoML"
        functionVersionInfo["ModelFunction"] = "UsePycaretModelByDatabaseRusult"
        functionVersionInfo["DataVersion"] = "P0_1_X"
        functionVersionInfo["ModelDesign"] = {
            "ModelDicts": [{
                "ModelFullName":functionVersionInfo["ModelFullName"],
                "ModelPackage":functionVersionInfo["ModelPackage"],
                "ModelName":functionVersionInfo["ModelName"],
                "ModelFileName":functionVersionInfo["ModelFileName"],
                "ModelStorageLocationPath": functionVersionInfo["ModelStorageLocationPath"],
                "ModelStorageLocation":functionVersionInfo["ModelStorageLocation"],
                "ModelStorageRemotePath":functionVersionInfo["ModelStorageRemotePath"],
                "ModelStorageRemote":functionVersionInfo["ModelStorageRemote"],
                "ModelResult": functionVersionInfo["ModelResult"],
            }]
        }
        functionVersionInfo["ModelParameter"] ={"TaskType": "Classification",}
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        resultDF = globalObjectDict["ResultArr"][0]
        ModelUseFunction.insertAnalysisData(
            functionInfo["Product"], functionInfo["Project"]
            , functionVersionInfo["OPSVersion"], functionVersionInfo["DataTime"].replace("-","")
            , resultDF, "IO"
        )
        return resultObject, globalObjectDict

    @classmethod
    def M0_11_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_11_X"])
        functionVersionInfo["Version"] = "M0_11_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_12_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_12_X"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_12_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_21_X(self, functionInfo):
        import os , copy
        import json
        from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_21_X"])
        opsVersionEntity = OPSVersionEntity()
        opsvEntity = opsVersionEntity.getOPSVersionByProductProjectOPSVersion(functionVersionInfo["Product"],functionVersionInfo["Project"],functionVersionInfo["Version"])
        parameterJson = json.loads(opsvEntity["parameterjson"])
        modelDicts = []
        if "ModelFullName" in parameterJson[functionVersionInfo["Function"]].keys():
            parameterJson[functionVersionInfo["Function"]]["UseModel"] = True
            modelDicts.append(parameterJson[functionVersionInfo["Function"]])
        return {}, {"ModelDictList":modelDicts}

    @classmethod
    def M0_22_X(self, functionInfo):
        import os , json , copy
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        from dotenv import load_dotenv

        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_22_X"])
        product = functionInfo["Product"]
        project = functionInfo["Project"]
        opsVersion = functionInfo["OPSVersion"]
        funcVersion = "M0_22_X"
        sql =  """
            SELECT
                AA.product
                , AA.project
                , AA.opsversion
                , BB.opsrecordid
                , CC.exefunction
                , CC.resultjson 
            FROM opsmanagement.opsversion AA 
            inner join opsmanagement.opsrecord BB on 1 = 1 
                and BB.opsversion = AA.opsversionid
                and bb.state = 'FINISH'
            inner join opsmanagement.opsdetail CC on 1 = 1 
                and CC.opsrecord = BB.opsrecordid
                and CC.exefunction = '[:Function]'
                and cc.state = 'FINISH'
                and CC.parameterjson::json ->> 'DataTime' >= to_char('[:DataLine]'::DATE - interval '6 day', 'YYYY-MM-DD')
                and CC.parameterjson::json ->> 'DataTime' <= to_char('[:DataLine]'::DATE - interval '0 day', 'YYYY-MM-DD')
            where 1 = 1
                and AA.product= '[:Product]'
                and AA.project = '[:Project]'
                and AA.opsversion = '[:Version]'
        """.replace("[:DataLine]",functionVersionInfo["DataTime"]) \
            .replace("[:DataNoLine]",functionVersionInfo["DataTime"].replace("-","")) \
            .replace("[:Product]",functionVersionInfo["Product"]) \
            .replace("[:Project]",functionVersionInfo["Project"]) \
            .replace("[:Version]",functionVersionInfo["Version"]) \
            .replace("[:Function]",functionVersionInfo["Function"])

        modelDicts = []
        resultjsonDF = postgresCtrl.searchSQL(sql)
        for index ,row in resultjsonDF.iterrows():
            resultjsonStr = row["resultjson"]
            resultjson = json.loads(resultjsonStr)
            for modelDist in resultjson["ModelDesign"]["ModelDicts"] :
                modelDist["DatabaseProduct"] = row["product"]
                modelDist["DatabaseProject"] = row["project"]
                modelDist["DatabaseOPSVersion"] = row["opsversion"]
                modelDist["DatabaseOPSRecord"] = row["opsrecordid"]
                modelDist["DatabaseFunction"] = row["exefunction"]
                modelDist["DatabaseModelName"] = modelDist["ModelName"]
                modelDist["MakeDataKeys"] = resultjson["MakeDataKeys"]
                modelDist["MakeDataInfo"] = resultjson["MakeDataInfo"]
                modelDist["UseModel"] = False
                modelDicts.append(modelDist)
        return {},{"ModelDictList":modelDicts}

    @classmethod
    def M0_23_X(self, functionInfo):
        import os , copy , pandas
        from dotenv import load_dotenv
        from package.common.common.osbasic.SSHCtrl import SSHCtrl
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        from pycaret.classification import predict_model
        from pycaret.classification import load_model

        load_dotenv(dotenv_path="env/ssh.env")
        sshCtrl = SSHCtrl(
            host=os.getenv("SSH_IP")
            , port=int(os.getenv("SSH_PORT"))
            , user=os.getenv("SSH_USER")
            , passwd=os.getenv("SSH_PASSWD")
        )

        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_23_X"])
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        verifyDataDF = globalObject["P0_29_X"]["ResultArr"][0]
        currentModelDictList = globalObject["M0_21_X"]["ModelDictList"]
        otherModelDictList = globalObject["M0_22_X"]["ModelDictList"]

        allModelDictList = currentModelDictList + otherModelDictList
        verifyModelDictArr = []

        for fvInfo in allModelDictList :
            try :
                tempDF = pandas.DataFrame()
                for key in fvInfo["MakeDataKeys"] :
                    tempDF[key] = verifyDataDF[key]
                for columnDict in fvInfo["MakeDataInfo"] :
                    dtDiffStr = "p" + str(abs(columnDict["DTDiff"])) if columnDict["DTDiff"] >= 0 else "n" + str(abs(columnDict["DTDiff"]))
                    for columnNumber in columnDict["ColumnNumbers"] :
                        columnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(columnDict["Product"], columnDict["Project"], dtDiffStr, str(columnNumber), columnDict["GFunc"],columnDict["Version"]))
                        tempDF[columnFullName] = verifyDataDF[columnFullName]

                fvInfo["ResultArr"] = [verifyDataDF]

                df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = ModelUseFunction.makeXYDataInfoAndColumnNames(fvInfo, {})

                modeldist = fvInfo

                os.makedirs(modeldist["ModelStorageLocationPath"]) if not os.path.isdir(modeldist["ModelStorageLocationPath"]) else None
                sshCtrl.downloadFile(modeldist['ModelStorageRemote'], modeldist['ModelStorageLocation'])
                bestmodel = load_model(modeldist['ModelStorageLocation'].replace(".pkl", ""))

                predictions = predict_model(bestmodel, data=df)
                predictions["prediction_label"] = predictions['prediction_label'] if "prediction_label" in predictions.columns else predictions['prediction_score']
                tp = int(((predictions[yColumnNames[0]] != 0) * (predictions['prediction_label'] != 0)).sum())
                fp = int(((predictions[yColumnNames[0]] != 0) * (predictions['prediction_label'] == 0)).sum())
                tn = int(((predictions[yColumnNames[0]] == 0) * (predictions['prediction_label'] == 0)).sum())
                fn = int(((predictions[yColumnNames[0]] == 0) * (predictions['prediction_label'] != 0)).sum())

                accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) != 0 else 0
                precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
                recall = (tp / (tp + fn)) if (tp + fn) != 0 else 0
                f1Score = ((2 * precision * recall) /(precision +recall) )if (precision +recall) != 0 else 0

                fvInfo["ResultArr"] = None

                verifyModelDict = copy.deepcopy(fvInfo)

                verifyModelDict["VerifyModelResult"] = {
                    "TP":float(tp), "FP":float(fp), "TN":float(tn), "FN":float(fn),
                    "Accuracy":float(accuracy), "Precision":float(precision),
                    "Recall":float(recall), "F1Score":float(f1Score),
                }

                verifyModelDictArr.append(verifyModelDict)
            except Exception as e :
                print(e)

        return {"ModelVerify":verifyModelDictArr}, {"ModelVerify":verifyModelDictArr}

    @classmethod
    def M0_24_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_24_X"])
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        verifyModelDictArr = globalObject["M0_23_X"]["ModelVerify"]

        bastVerifyModelDict = {}
        useVerifyModelDict = {}
        bastVerifyF1Score = 0
        useVerifyF1Score = 0

        for verifyModelDict in verifyModelDictArr :
            if verifyModelDict["UseModel"] :
                useVerifyModelDict = verifyModelDict
                useVerifyF1Score = verifyModelDict["VerifyModelResult"]["F1Score"]
            if verifyModelDict["VerifyModelResult"]["F1Score"] > bastVerifyF1Score and not verifyModelDict["UseModel"]:
                bastVerifyModelDict = verifyModelDict
                bastVerifyF1Score = verifyModelDict["VerifyModelResult"]["F1Score"]

        isModelUpdate = False
        if bastVerifyF1Score > (useVerifyF1Score + 0.01) :
            import json
            import OPSCommon as executeOPSCommon
            from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity
            opsVersionEntity = OPSVersionEntity()
            opsInfo = {"RunType": ["BuildOPS"]}
            opsInfo["Product"] = [functionVersionInfo["Product"]]
            opsInfo["Project"] = [functionVersionInfo["Project"]]
            opsInfo["OPSVersion"] = [functionVersionInfo["Version"]]
            opsvEntity = opsVersionEntity.getOPSVersionByProductProjectOPSVersion(opsInfo["Product"][0],opsInfo["Project"][0],opsInfo["OPSVersion"][0])
            opsInfo["OPSOrderJson"] = json.loads(opsvEntity["opsorderjson"])
            opsInfo["ParameterJson"] = json.loads(opsvEntity["parameterjson"])
            opsInfo["ParameterJson"][functionVersionInfo["Function"]] = bastVerifyModelDict

            opsInfo["ResultJson"] = {}
            executeOPSCommon.main(opsInfo)
            isModelUpdate = True

        resultObject = {
            "IsModelUpdate":isModelUpdate,
            "UseVerifyF1Score":useVerifyF1Score,
            "BastVerifyF1Score":bastVerifyF1Score,
            "UseVerifyModelDict":useVerifyModelDict,
            "BastVerifyModelDict":bastVerifyModelDict,
        }
        return resultObject, {}


