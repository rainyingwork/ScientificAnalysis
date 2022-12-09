import copy
import numpy , pandas
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

class ChartReport() :

    @classmethod
    def CR0_0_1(self, functionInfo):
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        from mpl_toolkits.mplot3d import Axes3D

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_11"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        # mainRecencyFlagDF = copy.deepcopy(globalObject["M0_0_1"]["ResultDFArr"][0])
        # mainFrequencyFlagDF = copy.deepcopy(globalObject["M0_0_2"]["ResultDFArr"][0])
        # mainMonetaryFlagDF = copy.deepcopy(globalObject["M0_0_3"]["ResultDFArr"][0])

        mainRecencyDF = copy.deepcopy(globalObject["M0_0_1"]["ResultDFArr"][1])
        mainFrequencyDF = copy.deepcopy(globalObject["M0_0_2"]["ResultDFArr"][1])
        mainMonetaryDF = copy.deepcopy(globalObject["M0_0_3"]["ResultDFArr"][1])

        mainAllDF = copy.deepcopy(globalObject["M0_0_11"]["MainAllDF"])
        mainPredictDF = copy.deepcopy(globalObject["M0_0_11"]["MainPredictDF"])


        for n_cluster in range(2, 7):
            kmeans = KMeans(n_clusters=n_cluster).fit(mainPredictDF)
            label = kmeans.labels_
            sil_coeff = silhouette_score(mainPredictDF, label, metric='euclidean')
            print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

        mainAllDF = mainAllDF.merge(mainRecencyDF[['CustomerID', 'YYMMNum']], on='CustomerID')
        mainAllDF = mainAllDF.merge(mainFrequencyDF[['CustomerID', 'InvoiceNo']], on='CustomerID')
        mainAllDF = mainAllDF.merge(mainMonetaryDF[['CustomerID', 'TotalPrice']], on='CustomerID')
        mainAllDF.rename(columns={'YYMMNum': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'},inplace=True)
        mainAllDF['Monetary'] = mainAllDF['Monetary'].round()
        print(mainAllDF.head())

        colors = ['purple', 'blue', 'green', 'gold']
        fig = plt.figure()
        fig.set_size_inches(12, 8)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(kmeans.n_clusters):
            df_cluster = mainAllDF[mainAllDF['Clusters'] == i]
            ax.scatter(df_cluster['RecencyFlag'], df_cluster['MonetaryFlag'], df_cluster['FrequencyFlag'], s=50,
                       label='Cluster' + str(i + 1))
        plt.legend()
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200,
                   marker='^', c='red', alpha=0.7, label='Centroids')
        plt.show()
        return {}, {"ResultDF": mainAllDF}