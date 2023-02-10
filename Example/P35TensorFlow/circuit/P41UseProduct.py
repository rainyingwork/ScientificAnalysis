
class UseProduct() :

    @classmethod
    def UP0_0_1(self, functionInfo):
        import copy
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_1"])
        functionVersionInfo["Version"] = "UP0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject["P0_0_1"]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"]["P0_0_1"]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"]["P0_0_1"]["MakeDataInfo"]
        model = globalObject["M0_0_1"]["Model"]

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = \
            modelUseFunction.makeXYDataInfoAndColumnNames(functionVersionInfo, {})

        trainDF, testDF = train_test_split(df, test_size=0.9)

        predictions = (model.predict(testDF[xColumnNames]) >= 0.5).astype("int32")
        tn, fp, fn, tp = confusion_matrix(testDF[yColumnNames], predictions.T[0]).ravel()
        modeldist = {}
        modeldist['ModelResult'] = {}
        modeldist['ModelResult']['TN'] = int(tn)
        modeldist['ModelResult']['FP'] = int(fp)
        modeldist['ModelResult']['FN'] = int(fn)
        modeldist['ModelResult']['TP'] = int(tp)
        modeldist['ModelResult']['Accuracy'] = tp / (tp + fp)
        modeldist['ModelResult']['Precision'] = tp / (tp + fn)
        modeldist['ModelResult']['Recall'] = (tp + tn) / (tp + tn + fp + fn)
        modeldist['ModelResult']['F1Score'] = 2 * modeldist['ModelResult']['Recall'] * modeldist['ModelResult']['Precision'] / (modeldist['ModelResult']['Recall'] + modeldist['ModelResult']['Precision'])

        return {"ModelDist": modeldist}, {}
