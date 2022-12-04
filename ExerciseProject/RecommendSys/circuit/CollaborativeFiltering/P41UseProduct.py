
class UseProduct() :

    @classmethod
    def UP0_2_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_2_1"])
        functionVersionInfo["Version"] = "UP0_2_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        mainPivotDF = copy.deepcopy(globalObject["M0_2_1"]["ResultDF"])
        movieName  = functionVersionInfo["MovieName"]
        topN = functionVersionInfo["TopN"] if "TopN" in functionVersionInfo.keys() else 5

        movieWatched = mainPivotDF[movieName]
        # 使用corrwith計算電影相關性
        similarityWithOtherMovies = mainPivotDF.corrwith(movieWatched)
        # 排序similarityWithOtherMovies
        similarityWithOtherMovies = similarityWithOtherMovies.sort_values(ascending=False)
        recommendResult = {}
        recommendResult["NameList"] = similarityWithOtherMovies.index[:topN].tolist()
        
        return {"RecommendResult":recommendResult}, {}

    @classmethod
    def UP0_2_2(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_2_2"])
        functionVersionInfo["Version"] = "UP0_2_2"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        mainPivotDF = copy.deepcopy(globalObject["M0_2_1"]["ResultDF"]).T
        userID  = functionVersionInfo["UserID"]
        topN = functionVersionInfo["TopN"] if "TopN" in functionVersionInfo.keys() else 5

        target_user = mainPivotDF[userID] # 10為客戶編號
        # 使用corrwith計算客戶相關性
        similarityWithOtherMovies = mainPivotDF.corrwith(target_user)
        # 排序similarityWithOtherMovies
        similarityWithOtherMovies = similarityWithOtherMovies.sort_values(ascending=False)
        recommendResult = {}
        recommendResult["NameList"] = similarityWithOtherMovies.index[:topN].tolist()

        return {"RecommendResult": recommendResult}, {}
