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

        def monthly(x):
            x = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            return datetime.datetime(x.year, x.month, 1)

        mainDF['BillMonth'] = mainDF['InvoiceDate'].apply(monthly)

        g = mainDF.groupby('CustomerID')['BillMonth']
        mainDF['CohortMonth'] = g.transform('min')
        def get_int(df, column):
            year = df[column].dt.year
            month = df[column].dt.month
            return year, month

        billYear, billMonth = get_int(mainDF, 'BillMonth')
        cohortYear, cohortMonth = get_int(mainDF, 'CohortMonth')
        diffYear = billYear - cohortYear
        diffMonth = billMonth - cohortMonth
        mainDF['Month_Index'] = diffYear * 12 + diffMonth + 1

        return {},{"ResultDF":mainDF}

