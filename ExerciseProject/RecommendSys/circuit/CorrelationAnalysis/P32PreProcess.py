
class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["R0_0_1"]
        mainDF = globalObjectFDict["ResultArr"][0]
        mainDF.columns = ["CustomerID", "Country", "InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice",
                          "InvoiceDate"]

        # 移除Description開頭為空的字符
        mainDF['Description'] = mainDF['Description'].str.strip()
        # 移除InvoiceNo為空的列(axis=0)並且回傳(inplace=False) --inplace=True為直接改
        mainDF = mainDF.dropna(subset=['InvoiceNo'], axis=0, inplace=False)
        # 移除InvoiceNo包含'C'(退貨)的發票
        mainDF = mainDF[~mainDF['InvoiceNo'].str.contains('C')]

        # 稀疏矩陣(Sparse Matrix)
        # unstack 針對樞紐的最後幾個項目轉為欄位
        basketSMDF = mainDF.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(
            0).set_index('InvoiceNo')

        # 購物籃分析不考慮數量，將數量 >0 的值全部轉為 1。
        basketSMDF = basketSMDF.applymap(lambda x: 1 if x > 0 else 0)

        return {}, {"ResultDF": basketSMDF}

