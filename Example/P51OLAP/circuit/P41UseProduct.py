from package.common.common.osbasic.BaseFunction import timethis

class UseProduct() :

    @classmethod
    @timethis
    def UP0_0_1(self, functionInfo):
        import polars
        import copy
        import json
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
        preProcessFunction = PreProcessFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["UP0_0_1"])
        functionVersionInfo["Version"] = "UP0_0_1"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        oriDF = globalObject["R0_0_1"]["oriDF"]

        # 可以使用一般方式來增加過濾相關資料，非常快速

        resultDF = oriDF.filter(
            (polars.col('Registration State') == "NY") &
            (polars.col('Plate Type') == "PAS") &
            (polars.col('Plate ID') == "GZH7067")
        )

        # 可以使用SQL方式來快速過濾相關資料，但跟SQL方式不同的是
        # 1.不能有子查詢,WITH 相關方式皆不支援使用 請多寫幾層SQL
        # 2.Table無法重新命名,請在外面定義好
        # 3.JOIN XXX ON 1 = 1 使用 1 = 1 會有錯誤 請勿使用
        # 4.表格常常可能會有大寫或空格 請使用""包起來
        # 5.函數有些不支援 可以搭配Python函數使用

        resultDF = polars.SQLContext(AA=oriDF).execute("""
            SELECT
                AA."Summons Number" as sn
                , AA."Plate ID" as id 
            FROM AA
            WHERE 1 = 1
                AND AA."Registration State" = 'NY'
                AND AA."Plate Type" = 'PAS'
                AND AA."Plate ID" = 'GZH7067'
        """,eager=True)

        return {"result":resultDF.to_pandas().to_dict(orient='records')}, {}
