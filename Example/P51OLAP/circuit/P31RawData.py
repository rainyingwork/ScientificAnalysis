from package.common.common.osbasic.BaseFunction import timethis
class RawData() :

    @classmethod
    @timethis
    def R0_0_1(self, functionInfo):
        # R0_0_1 0:00:01.771999
        import polars
        oriDF = polars.read_csv("common/common/file/data/csv/archive/Parking_Violations_Issued_-_Fiscal_Year_2017.csv")
        return {} ,{"oriDF":oriDF}

    @classmethod
    @timethis
    def R0_0_2(self, functionInfo):
        # R0_0_2 0:00:29.668641
        import pandas
        oriDF = pandas.read_csv("common/common/file/data/csv/archive/Parking_Violations_Issued_-_Fiscal_Year_2017.csv")
        return {}, {"oriDF":oriDF}

    @classmethod
    @timethis
    def R0_0_3(self, functionInfo):
        return {}, {}