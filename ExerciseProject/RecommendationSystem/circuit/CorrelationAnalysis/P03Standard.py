
class Standard () :

    @classmethod
    def S0_0_1(self, functionInfo):
        import pandas as pd
        df = pd.read_excel('ExerciseProject/RecommendationSystem/file/data/Online Retail.xlsx')
        print(df.info())
        print(df.head())
        return {} ,{}