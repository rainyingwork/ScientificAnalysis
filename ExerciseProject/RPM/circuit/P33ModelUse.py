import copy
import numpy , pandas
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl

class ModelUse() :
    
    @classmethod
    def M0_0_1(self, functionInfo):
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_0_1"]
        mainDF = copy.deepcopy(globalObjectFDict["ResultDF"])

        # ["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]
        mainDF['YYMMNum'] = mainDF['InvoiceDate'].astype(numpy.str).str.slice(0, 8).str.replace('-', '')
        mainDF['YYMMNum'] = pandas.to_numeric(mainDF['YYMMNum'], errors='coerce')

        def makeRecencyLevel(row):
            if row['YYMMNum'] > 201110:
                val = 5
            elif row['YYMMNum'] > 201108:
                val = 4
            elif row['YYMMNum'] > 201106:
                val = 3
            elif row['YYMMNum'] > 201104:
                val = 2
            else:
                val = 1
            return val

        mainRecencyDF = mainDF[['CustomerID', 'YYMMNum']].drop_duplicates()
        mainRecencyDF['RecencyFlag'] = mainRecencyDF.apply(makeRecencyLevel, axis=1)
        mainFlagDF = mainRecencyDF.groupby('CustomerID', as_index=False)['RecencyFlag'].max()
        return {} ,{"ResultDFArr": [mainFlagDF,mainRecencyDF]}

    @classmethod
    def M0_0_2(self, functionInfo):
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_2"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_0_1"]
        mainDF = copy.deepcopy(globalObjectFDict["ResultDF"])

        # ["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]
        mainDF = mainDF[['CustomerID' , 'InvoiceNo']].drop_duplicates()
        mainFrequencyDF = mainDF.groupby(['CustomerID'])['InvoiceNo'].aggregate('count').reset_index().sort_values('InvoiceNo', ascending=False, axis=0)

        # 找出購買頻率5分箱數字，保留分析道事前分析報表
        # unique_invoice = mainFrequencyDF[['InvoiceNo']]
        # unique_invoice['Freqency_Band'] = pandas.qcut(unique_invoice['InvoiceNo'], 5)
        # unique_invoice = unique_invoice[['Freqency_Band']].drop_duplicates()
        # print(unique_invoice)

        def makeFrequencyLevel(row):
            if row['InvoiceNo'] < 1:
                val = 1
            elif row['InvoiceNo'] <= 2:
                val = 2
            elif row['InvoiceNo'] <= 3:
                val = 3
            elif row['InvoiceNo'] <= 6:
                val = 4
            else:
                val = 5
            return val

        mainFrequencyDF['FrequencyFlag'] = mainFrequencyDF.apply(makeFrequencyLevel, axis=1)
        mainFlagDF = mainFrequencyDF[['CustomerID','FrequencyFlag']]
        return {}, {"ResultDFArr": [mainFlagDF,mainFrequencyDF]}

    @classmethod
    def M0_0_3(self, functionInfo):
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_3"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_0_1"]
        mainDF = copy.deepcopy(globalObjectFDict["ResultDF"])

        # 根據這期間的購買金額去算出總金額
        mainMonetaryDF = mainDF.groupby(['CustomerID'])['TotalPrice'].aggregate('sum').reset_index().sort_values('TotalPrice', ascending=False)

        # 找出購買金額的5分箱數字，保留分析道事前分析報表
        # unique_price = Cust_monetary[['Total_Price']].drop_duplicates()
        # unique_price = unique_price[unique_price['Total_Price'] > 0]
        # unique_price['monetary_Band'] = pandas.qcut(unique_price['Total_Price'], 5)
        # unique_price = unique_price[['monetary_Band']].drop_duplicates()
        # print(unique_price)

        def makeMonetaryLevel(row):
            if row['TotalPrice'] <= 243:
                val = 1
            elif row['TotalPrice'] > 243 and row['TotalPrice'] <= 463:
                val = 2
            elif row['TotalPrice'] > 463 and row['TotalPrice'] <= 892:
                val = 3
            elif row['TotalPrice'] > 892 and row['TotalPrice'] <= 1932:
                val = 4
            else:
                val = 5
            return val

        mainMonetaryDF['MonetaryFlag'] = mainMonetaryDF.apply(makeMonetaryLevel, axis=1)
        mainFlagDF = mainMonetaryDF[['CustomerID','MonetaryFlag']]
        return {}, {"ResultDFArr": [mainFlagDF,mainMonetaryDF]}

    @classmethod
    def M0_0_11(self, functionInfo):
        from sklearn.cluster import KMeans
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_11"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        mainRecencyDF = copy.deepcopy(globalObject["M0_0_1"]["ResultDFArr"][0])
        mainFrequencyDF = copy.deepcopy(globalObject["M0_0_2"]["ResultDFArr"][0])
        mainMonetaryDF = copy.deepcopy(globalObject["M0_0_3"]["ResultDFArr"][0])

        mainAllDF = pandas.merge(mainRecencyDF, mainFrequencyDF[['CustomerID', 'FrequencyFlag']], on=['CustomerID'], how='left')
        mainAllDF = pandas.merge(mainAllDF, mainMonetaryDF[['CustomerID', 'MonetaryFlag']], on=['CustomerID'], how='left')

        # def findwcss(df):
        #     from sklearn.cluster import KMeans
        #
        #     wcss = []
        #     for i in range(1, 11):
        #         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        #         kmeans.fit(df)
        #         wcss.append(kmeans.inertia_)
        #
        #     import matplotlib.pyplot as plt
        #     plt.plot(range(1, 11), wcss, marker='o')
        #     plt.title('Elbow graph')
        #     plt.xlabel('Cluster number')
        #     plt.ylabel('WCSS')
        #     plt.show()
        # 尋找損失函數wcss 這邊先備註 免得每次執行浪費太多效能
        # findwcss(Cust_All)


        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
        mainPredictDF = mainAllDF.drop(mainAllDF.columns[0], axis=1)
        mainAllDF['Clusters'] = kmeans.fit_predict(mainPredictDF)

        return {}, {"MainAllDF": mainAllDF , "MainPredictDF":mainPredictDF}

