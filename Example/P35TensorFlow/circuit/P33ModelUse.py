
class ModelUse() :
    
    @classmethod
    def M0_0_1(self, functionInfo):
        import copy
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["ResultArr"]
        functionVersionInfo["MakeDataKeys"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataKeys"]
        functionVersionInfo["MakeDataInfo"] = functionInfo["ResultJson"][functionInfo["ParameterJson"]["M0_0_1"]["DataVersion"]]["MakeDataInfo"]

        df , yMakeDataInfoArr, xMakeDataInfoArr , commonColumnNames, yColumnNames, xColumnNames = \
            modelUseFunction.makeXYDataInfoAndColumnNames(functionVersionInfo, {})

        trainDF, testDF = train_test_split(df, test_size=0.2)

        import tensorflow as tf

        tf.random.set_seed(42)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(lr=0.03),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        model.fit(trainDF[xColumnNames], trainDF[yColumnNames], epochs=100)
        model.summary()
        predictions = (model.predict(testDF[xColumnNames]) >= 0.5).astype("int32")
        print(predictions)
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
        return {"ModelDist":modeldist}, {"Model":model}