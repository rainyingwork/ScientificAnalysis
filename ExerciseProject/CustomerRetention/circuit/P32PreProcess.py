import copy
from package.common.osbasic.GainObjectCtrl import GainObjectCtrl


class PreProcess() :

    @classmethod
    def P0_0_1(self, functionInfo):
        import datetime

        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["P0_0_1"])
        functionVersionInfo["Version"] = "P0_0_1"
        globalObjectFDict = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])["R0_0_1"]
        mainDF = globalObjectFDict["ResultArr"][0]
        mainDF.columns = ["CustomerID","InvoiceNo","InvoiceDate","TotalPrice"]

        # 資料整理過程
        mainDF['BillMonth'] = mainDF['InvoiceDate'].apply(lambda x : datetime.datetime(x.year, x.month, 1))
        groupDF = mainDF.groupby('CustomerID')['BillMonth']
        mainDF['StartMonth'] = groupDF.transform('min')
        billYear = mainDF['BillMonth'].dt.year
        billMonth = mainDF['BillMonth'].dt.month
        cohortYear = mainDF['StartMonth'].dt.year
        cohortMonth = mainDF['StartMonth'].dt.month
        diffYear = billYear - cohortYear
        diffMonth = billMonth - cohortMonth
        mainDF['MonthIndex'] = diffYear * 12 + diffMonth + 1

        return {},{"ResultDF":mainDF}

