
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
        modeldict = {}
        modeldict['ModelResult'] = {}
        modeldict['ModelResult']['TN'] = int(tn)
        modeldict['ModelResult']['FP'] = int(fp)
        modeldict['ModelResult']['FN'] = int(fn)
        modeldict['ModelResult']['TP'] = int(tp)
        modeldict['ModelResult']['Accuracy'] = tp / (tp + fp)
        modeldict['ModelResult']['Precision'] = tp / (tp + fn)
        modeldict['ModelResult']['Recall'] = (tp + tn) / (tp + tn + fp + fn)
        modeldict['ModelResult']['F1Score'] = 2 * modeldict['ModelResult']['Recall'] * modeldict['ModelResult']['Precision'] / (modeldict['ModelResult']['Recall'] + modeldict['ModelResult']['Precision'])
        return {"ModelDict":modeldict}, {"Model":model}