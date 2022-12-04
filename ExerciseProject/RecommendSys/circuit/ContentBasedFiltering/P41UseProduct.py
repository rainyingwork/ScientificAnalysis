
class UseProduct() :

    def UP0_1_1(self, functionInfo):
        import copy, pandas
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_1_1"])
        functionVersionInfo["Version"] = "UP0_1_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        mainKeysDF = globalObject["M0_1_2"]["ResultDF"]
        CVMCS = globalObject["M0_1_2"]["CVMCS"]
        movieName = functionVersionInfo["MovieName"]

        indices = pandas.Series(mainKeysDF.index, index=mainKeysDF['Title'])
        def makeRecommendMovie(title, indices, CVMCS , n=5):
            # 找到該電影的index編號
            if title not in indices.index:
                return
            else:
                idx = indices[title]
            # 針對該index欄位的相似度做排序分數
            scores = pandas.Series(CVMCS[idx]).sort_values(ascending=False)
            # 撈取相關topIndexs編號
            topIndexs = list(scores.iloc[1:n].index)
            return mainKeysDF['MovieID'].iloc[topIndexs].tolist() , mainKeysDF['Title'].iloc[topIndexs].tolist()
        recommendResult ={}
        recommendResult["IDList"] , recommendResult["NameList" ] = makeRecommendMovie(movieName, indices, CVMCS )
        return {"RecommendResult":recommendResult}, {}
