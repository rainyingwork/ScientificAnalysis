
class ModelUse() :
    
    @classmethod
    def M0_0_1(self, functionInfo):
        import copy
        from package.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["M0_0_1"])
        functionVersionInfo["Version"] = "M0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["P0_0_1"]
        basketSMDF = globalObjectFDict["ResultDF"]
        # 使用 apriori 做出相關的關聯組合表
        # 相關 apriori 使用方式可以參考 https://ithelp.ithome.com.tw/articles/10218530
        basketSMFilterDF = apriori(basketSMDF, min_support=0.06, use_colnames=True)
        resultDF = association_rules(basketSMFilterDF, metric="lift", min_threshold=1)
        resultDF.sort_values('lift', ascending=False)
        return {},{"ResultDF":basketSMDF}
