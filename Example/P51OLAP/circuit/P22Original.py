import os
import pandas
import polars
from package.common.common.osbasic.BaseFunction import timethis
import copy
from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
from package.artificialintelligence.common.preprocess.PreProcessFunction import PreProcessFunction
class Original():

    @classmethod
    @timethis
    def O0_0_1(self, functionInfo):
        import polars
        import random
        num_rows = 1000000000
        numbers = list(range(1, num_rows + 1))
        # 创建一个包含这些数字的Polars DataFrame
        polarsDF = polars.DataFrame({
            'numbers': numbers
        })
        return {"result": "OK"}, {'polarsDF':polarsDF}

    @classmethod
    @timethis
    def O0_0_2(self, functionInfo):
        import polars
        import random
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        polarsDF = globalObject["O0_0_1"]["polarsDF"]
        num_rows = 1000000000
        random_bit_data = [random.choice([True, False]) for _ in range(num_rows)]
        # 创建一个包含这些数字的Polars DataFrame
        polarsDF = polars.DataFrame({
            'random_bit_0': random_bit_data
        })
        return {"result": "OK"}, {'polarsDF': polarsDF}

    @classmethod
    @timethis
    def O0_0_3(self, functionInfo):
        import polars
        import random
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        polarsDF = globalObject["O0_0_1"]["polarsDF"]
        polarsRandomDF = globalObject["O0_0_2"]["polarsDF"]
        polarsDF = polarsDF.hstack(polarsRandomDF)
        for i in range(1, 400):
            print(i)
            polarsDF = polarsDF.with_columns((polars.col("random_bit_0")).alias("random_bit_" + str(i)))
        print(polarsDF.dtypes)
        polarsDF.write_parquet("Example/P51OLAP/file/data/polars_test.parquet")
        return {"result": "OK"}, {}


    @classmethod
    @timethis
    def O1_0_1(self, functionInfo):
        import polars
        import random
        polars1DF = polars .read_parquet("Example/P51OLAP/file/data/polars_test.parquet")
        print("polars1DF")
        # polars2DF = polars .read_parquet("Example/P51OLAP/file/data/polars_test.parquet")
        # print("polars2DF")
        # polars3DF = polars .read_parquet("Example/P51OLAP/file/data/polars_test.parquet")
        # print("polars3DF")
        # polars4DF = polars .read_parquet("Example/P51OLAP/file/data/polars_test.parquet")
        # print("polars4DF")
        return {"result": "OK"}, {"polars1DF":polars1DF}

    @classmethod
    @timethis
    def O1_0_2(self, functionInfo):
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        polars1DF = globalObject["O1_0_1"]["polars1DF"]
        sql = ""
        for i in range(0, 400):
            sql = sql + "            AND random_bit_" + str(i) + " = True \n"
        polarsS = polars.SQLContext(AA=polars1DF).execute("""
           SELECT count(*) FROM AA
           WHERE 1 = 1
             [:WHERE]
        """.replace("[:WHERE]", sql),eager=True)
        print("""
           SELECT count(*) FROM AA
           WHERE 1 = 1
            [:WHERE]      
        """.replace("[:WHERE]", sql))

        print(polarsS)
        # 128G資料 一兆個參數 bytes 大約22.6分可以跑完
        # 但建議負載到達50% 就好 也就是 5000億參數(bit) 11.3分鐘
        # 正常來說float64 會佔到8個bytes 也就是64位元 所以最多大概 78億參數 全部瀏覽過一遍大概需要 100秒\
        # 正常來說會有 100Tag * 10 萬筆資料 大約 1000萬筆 最慢大概 0.1 s + python 運作時間
        return {"result": "OK"}, {}


