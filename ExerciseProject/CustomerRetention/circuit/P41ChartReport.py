import copy
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

class ChartReport() :

    @classmethod
    def CR0_0_1(self, functionInfo):
        import pandas
        import matplotlib.pyplot as plt
        import seaborn as sns

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["CR0_0_1"])
        functionVersionInfo["Version"] = "CR0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])

        mainDF = copy.deepcopy(globalObject["P0_0_1"]["ResultDF"])
        groupDF = mainDF.groupby(['StartMonth', 'MonthIndex'])
        cohortData = groupDF['CustomerID'].apply(pandas.Series.nunique).reset_index()
        cohortCounts = cohortData.pivot(index='StartMonth', columns='MonthIndex', values='CustomerID')
        cohortSizes = cohortCounts.iloc[:, 0]
        retention = cohortCounts.divide(cohortSizes, axis=0) * 100

        month_list = ["Dec", "Jan", "Feb", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plt.figure(figsize=(20, 10))
        plt.title('Retention by Monthly Cohorts')
        sns.heatmap(retention.round(2), annot=True
                    , cmap="Blues", vmax=list(retention.max().sort_values(ascending=False))[1] + 3
                    , fmt='.1f', linewidth=0.3, yticklabels=month_list)
        plt.show()
        return {}, {"ResultDF": mainDF}