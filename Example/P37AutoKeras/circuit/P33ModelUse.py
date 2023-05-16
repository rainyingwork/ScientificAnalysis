
class ModelUse() :

    @classmethod
    def M0_0_1(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_2(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_2"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_3(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        modelUseFunction = ModelUseFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_3"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_3"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = modelUseFunction.executionFunctionByFunctionType(functionVersionInfo)
        ModelUseFunction.insertOverwriteAnalysisData("Example","P37AutoKeras","V0_0_3","20220101",globalObjectDict["ResultArr"][0], useType= 'IO')
        return resultObject, globalObjectDict

    @classmethod
    def M0_0_4(self, functionInfo):
        import copy
        import pandas
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.modeluse.ModelUseFunction import ModelUseFunction
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_4"])
        functionVersionInfo["OPSVersionId"] = functionInfo["OPSVersionId"]
        functionVersionInfo["OPSRecordId"] = functionInfo["OPSRecordId"]
        functionVersionInfo["Product"] = functionInfo["Product"]
        functionVersionInfo["Project"] = functionInfo["Project"]
        functionVersionInfo["OPSVersion"] = functionInfo["OPSVersion"]
        functionVersionInfo["Version"] = "M0_0_3"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]

        df, yMakeDataInfoArr, xMakeDataInfoArr, commonColumnNames, yColumnNames, xColumnNames = ModelUseFunction().makeXYDataInfoAndColumnNames(functionVersionInfo, {})

        from pycaret.clustering import setup , create_model , assign_model

        exp_clu = setup(data=df[xColumnNames])

        for modelName in ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch", "kmodes"]:
            try:
                model = create_model(modelName)
                result = assign_model(model)
                new_df = df[yColumnNames[0]].apply(lambda x: str(int(x))).to_frame()
                new_df['Cluster'] = result['Cluster'].apply(lambda x: str(int(x.replace('Cluster ', ''))))
                new_df['Value'] = 1
                pivot_table = pandas.pivot_table(new_df, values='Value', index=yColumnNames,columns=['Cluster'], aggfunc='sum')
                print(pivot_table)
            except:
                pass

        return {}, {}