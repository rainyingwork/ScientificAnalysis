
class ModelUse() :

    @classmethod
    def M0_2_1(self, functionInfo):
        # import copy , pandas
        # from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        # from rake_nltk import Rake
        #
        # functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_2_1"])
        # functionVersionInfo["Version"] = "M0_2_1"
        # globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_2_1"]
        # mainDF = globalObjectFDict["ResultDF"]
        #
        # def getSentenceKeywords(sentence):
        #     # 關鍵字提取模型 rake_nltk
        #     rake = Rake()
        #     # 將句子丟入已經訓練好的 rake_nltk
        #     rake.extract_keywords_from_text(sentence)
        #     # 提取該句關鍵字並轉化成List
        #     sentenceKeywordList = list(rake.get_word_degrees().keys())
        #     return sentenceKeywordList
        #
        # mainDF['PlotWords'] = ''
        # mainDF['PlotWords'] = mainDF['Overview'].apply(getSentenceKeywords)
        # mainKeysDF = pandas.DataFrame()
        # mainKeysDF['MovieID'] = mainDF['MovieID']
        # mainKeysDF['Title'] = mainDF['OriginalTitle']
        # mainKeysDF['AllKeywords'] = ''
        #
        # def makeWordBags(x):
        #     return (' '.join(x['Genres']) + ' ' + ' '.join(x['Keywords']) + ' ' + ' '.join(x['Cast']) +' ' + ' '.join(x['Director']) + ' ' + ' '.join(x['PlotWords']))
        #
        # # 製作所有的關鍵字變成一整個字袋
        # mainKeysDF['AllKeywords'] = mainDF.apply(makeWordBags, axis=1)
        #
        # return {}, {"ResultDF": mainKeysDF}
        return {}, {}

    @classmethod
    def M0_2_2(self, functionInfo):
        # import copy , pandas
        # from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        # from sklearn.metrics.pairwise import cosine_similarity
        # from sklearn.feature_extraction.text import CountVectorizer
        #
        # functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_2_2"])
        # functionVersionInfo["Version"] = "M0_2_2"
        # globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        # mainKeysDF = globalObject["M0_2_1"]["ResultDF"]
        #
        # # 詞袋模型 CountVectorizer
        # countVectorizer = CountVectorizer()
        # # 將詞袋丟進詞袋模型CountVectorizer產生可以用cvMatrix,可以用cvMatrix.toarray()來查看結果
        # cvMatrix = countVectorizer.fit_transform(mainKeysDF['AllKeywords'])
        # # 使用餘弦相似性CosineSimilarity來查看各物品的相似程度 CVMCS-CountVectorizerMatrixCosineSimilarity
        # CVMCS = cosine_similarity(cvMatrix, cvMatrix)
        # return {}, {"ResultDF": mainKeysDF, "CVMCS":CVMCS}
        return {}, {}
