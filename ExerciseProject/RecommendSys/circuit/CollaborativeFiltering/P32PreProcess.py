
class PreProcess() :

    @classmethod
    def P0_2_1(self, functionInfo):
        # import copy , numpy
        # from ast import literal_eval
        # from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        # functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_2_1"])
        # functionVersionInfo["Version"] = "P0_2_1"
        # globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        # movieMainDF = globalObject["R0_2_1"]["ResultArr"][0]
        # movieDetailDF = globalObject["R0_2_2"]["ResultArr"][0]
        # movieMainDF.columns = ['MovieID', 'Title', 'Cast', 'Crew']
        # movieDetailDF.columns = [
        #     'MovieID'
        #     , 'Genres', 'Keywords', 'ProductionCompanies', 'ProductionCountries', 'SpokenLanguages'
        #     , 'Title', 'OriginalTitle', 'OriginalLanguage'
        #     , 'Tagline', 'Homepage', 'Overview', 'Status'
        #     , 'Budget', 'Revenue'
        #     , 'VoteAverage', 'VoteCount', 'Popularity', 'Runtime'
        #     , 'ReleaseDate'
        # ]
        #
        # movieAllDF = movieMainDF.merge(movieDetailDF, on='MovieID')
        # mainDF = movieAllDF[['MovieID', 'OriginalTitle', 'OriginalLanguage', 'Overview', 'Genres', 'Keywords', 'Cast', 'Crew']]
        #
        # # 空值處理
        # mainDF['Overview'] = mainDF['Overview'].fillna('')
        #
        # # JSON 轉為陣列Dict
        # mainDF['Genres'] = movieAllDF['Genres'].apply(literal_eval)
        # mainDF['Keywords'] = movieAllDF['Keywords'].apply(literal_eval)
        # mainDF['Cast'] = movieAllDF['Cast'].apply(literal_eval)
        # mainDF['Crew'] = movieAllDF['Crew'].apply(literal_eval)
        #
        # def getDirector(crews):
        #     for crew in crews:
        #         if crew['job'] == 'Director':
        #             return crew['name']
        #     return numpy.nan
        #
        # # 陣列Dict 轉 陣列Str
        # mainDF['Genres'] = mainDF['Genres'].apply(lambda x : ([d['name'] for d in x]))
        # mainDF['Keywords'] = mainDF['Keywords'].apply(lambda x :([d['name'] for d in x][:3] if len([d['name'] for d in x]) > 5 else [d['name'] for d in x]))
        # mainDF['Cast'] = mainDF['Cast'].apply(lambda x :([d['name'] for d in x][:3] if len([d['name'] for d in x]) > 3 else [d['name'] for d in x]))
        # mainDF['Director'] = mainDF['Crew'].apply(getDirector)
        # mainDF = mainDF.drop('Crew', axis=1)
        #
        # def cleanSpace(x):
        #     if isinstance(x, list):
        #         return [i.lower().replace(" ", "") for i in x]
        #     else:
        #         if isinstance(x, str):
        #             return x.lower().replace(" ", "")
        #         else:
        #             return ''
        #
        # # 陣列Str去除所有空白字元與轉小寫
        # features = ['Genres', 'Keywords', 'Cast', 'Director']
        # for feature in features:
        #     mainDF[feature] = mainDF[feature].apply(cleanSpace)
        #
        # return {}, {"ResultDF": mainDF}
        return {}, {}

