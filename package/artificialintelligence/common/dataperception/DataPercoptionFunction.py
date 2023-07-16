import os
from dotenv import load_dotenv
from package.common.common.database.PostgresCtrl import PostgresCtrl
from package.artificialintelligence.common.dataperception.DataPercoptionTool import DataPercoptionTool

class DataPercoptionFunction(DataPercoptionTool):

    def __init__(self):
        pass

    @classmethod
    def executionFunctionByFunctionType(self, functionVersionInfo):
        resultDict = {}
        globalObjectDict = {}
        if functionVersionInfo['FunctionType'] == "MakeDataPercoption":
            otherInfo = self.dpDataPercoption(functionVersionInfo)
            globalObjectDict['ResultArr'] = otherInfo["DFArr"]
        elif functionVersionInfo['FunctionType'] == "CompareDataPercoptionByDT":
            otherInfo = self.dpCompareDataPercoptionByDT(functionVersionInfo)
        elif functionVersionInfo['FunctionType'] == "CompareDataPercoptionByTableName":
            otherInfo = self.dpCompareDataPercoptionByTableName(functionVersionInfo)
        resultDict['Result'] = "OK"
        return resultDict , globalObjectDict

    # ================================================== MainFunction ==================================================

    @classmethod
    def dpDataPercoption(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.makeDataPercoption(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def dpCompareDataPercoptionByDT(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.compareDataPercoptionByDT(fvInfo, otherInfo)
        return otherInfo

    @classmethod
    def dpCompareDataPercoptionByTableName(self, fvInfo):
        otherInfo = {}
        otherInfo["DFArr"] = self.compareDataPercoptionByTableName(fvInfo, otherInfo)
        return otherInfo

    # ================================================= DataPercoption =================================================

    @classmethod
    def makeDataPercoption(self,fvInfo, otherInfo):
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        tableNameArr = fvInfo["TableName"] if type(fvInfo["TableName"]).__name__ == "list" else [fvInfo["TableName"]]
        dataTimeArr = fvInfo["DataTime"] if type(fvInfo["DataTime"]).__name__ == "list" else [fvInfo["DataTime"]]

        for tableName in tableNameArr:
            for dataTime in dataTimeArr:
                eval(f"exec('from {fvInfo['Product']}.{fvInfo['Project']}.info.InfoMain import InfoMain')")
                infoMain = eval(f"InfoMain()")
                columnInfoMap = eval(f"infoMain.getInfo_{tableName}({fvInfo})")
                percopDetailAllSQL = ""
                percopColumnALLSQLs = []
                percopColumnALLSQLs.append("""'all_data_datacount' , "all_data_datacount" """)
                for columnKey in columnInfoMap.keys():
                    columnInfo = columnInfoMap[columnKey]
                    columnInfo['columnname'] = columnKey
                    if 'checkfuncs' not in columnInfo.keys():
                        continue
                    cfArr = []
                    for checkfunc in columnInfo['checkfuncs']:
                        percopColumnSQL, percopDetailSQL = eval(f"self.makePercoption{checkfunc}SQL({columnInfo})")
                        percopDetailAllSQL = percopDetailAllSQL + "\n        , {} ".format(percopDetailSQL)
                        cfArr.append(percopColumnSQL)
                    percopColumnALLSQLs.append('\n                    , '.join(cfArr))

                percopSingleInitSQL = DataPercoptionTool().makeBasicPercoptionDetailSQL(fvInfo)

                percopColumnALLSQLArr = []
                for percopColumnALLSQL in percopColumnALLSQLs:
                    percopSingleSQL = percopSingleInitSQL.replace("[:PERCOP_COLUMN]", percopColumnALLSQL)
                    percopColumnALLSQLArr.append(percopSingleSQL)

                exeSQLs = DataPercoptionTool().makeBasicPercoptionSQL(fvInfo).replace("[:Real_TABLE_NAME]",fvInfo["RealTableName"])
                exeSQLs = exeSQLs.replace("[:PRODUCT]", fvInfo["Product"])
                exeSQLs = exeSQLs.replace("[:PROJECT]", fvInfo["Project"])
                exeSQLs = exeSQLs.replace("[:TABLENAME]", tableName)
                exeSQLs = exeSQLs.replace("[:DateNoLine]", dataTime.replace("-", ""))
                exeSQLs = exeSQLs.replace("[:PERCOP_COLUMN_VALSE_SELECT]", "null")
                exeSQLs = exeSQLs.replace("[:PERCOP_COLUMN_VALSE_GROUP]", "")
                exeSQLs = exeSQLs.replace("[:PERCEP_CYCLE]", fvInfo["PercepCycle"])
                exeSQLs = exeSQLs.replace("[:PERCOP_DETAIL]", percopDetailAllSQL)
                exeSQLs = exeSQLs.replace("[:PERCOP_SQL]", "UNION ALL ".join(percopColumnALLSQLArr))

                exeSQLArr = exeSQLs.split(";")[:-1]

                for exeSQL in exeSQLArr:
                    postgresCtrl.executeSQL(exeSQL)

        return []

    @classmethod
    def compareDataPercoptionByDT(self, fvInfo, otherInfo):
        import os
        import pandas
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import seaborn as sns

        pandas.set_option('display.float_format', '{:,.2f}'.format)

        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        searchInitSQL = """
        select 
        	AA.datatable
        	, AA.product
        	, AA.project
        	, AA.datetime
        	, AA.version
        	, AA.columnname
        	, AA.columnvalue
        	, AA.percepfunc
        	, AA.percepcycle
        	, AA.percepvalue
        	, AA.percepstate
        	, AA.percepmemojson 
        	, (AA.percepmemojson::json->>'diff')::DOUBLE PRECISION as diff
        	, (AA.percepmemojson::json->>'log10diff')::DOUBLE PRECISION as log10diff
        from dataperception.dpmain AA
        WHERE 1 = 1 
            and AA.datatable = '[:Real_TABLE_NAME]'
            and AA.Product = '[:PRODUCT]'
            and AA.Project = '[:PROJECT]'
            and AA.version = '[:TABLENAME]'
            and case 
                    when '[:PERCEP_CYCLE]' = '1D' then AA.datetime <= to_char((date '[:DateNoLine]' + integer '-0'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1W' then AA.datetime <= to_char((date '[:DateNoLine]' + integer '-0'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1M' then AA.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '0 month') + interval '1 month' - interval '1 day','yyyyMMdd')
                end
            and case 
                    when '[:PERCEP_CYCLE]' = '1D' then AA.datetime >= to_char((date '[:DateNoLine]' + integer '-29'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1W' then AA.datetime >= to_char((date '[:DateNoLine]' + integer '-203'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1M' then AA.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '23 month') ,'yyyyMMdd')
                end
            ; 
        """

        searchSQL = searchInitSQL.replace("[:Real_TABLE_NAME]", fvInfo["RealTableName"])
        searchSQL = searchSQL.replace("[:PRODUCT]", fvInfo["Product"])
        searchSQL = searchSQL.replace("[:PROJECT]", fvInfo["Project"])
        searchSQL = searchSQL.replace("[:TABLENAME]", fvInfo["Tablename"])
        searchSQL = searchSQL.replace("[:DateNoLine]", fvInfo["DataTime"].replace("-", ""))
        searchSQL = searchSQL.replace("[:PERCEP_CYCLE]", fvInfo["PercepCycle"])

        df = postgresCtrl.searchSQL(searchSQL)

        df["YYYY"] = df["datetime"].str[0:4]
        df["MM"] = df["datetime"].str[4:6]
        df["DD"] = df["datetime"].str[6:8]


        sortColumnnameArr = [
            "all_data"
            , "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "string_001", "string_002", "string_003", "string_004", "string_005"
            , "string_006", "string_007", "string_008", "string_009", "string_010"
            , "integer_001", "integer_002", "integer_003", "integer_004", "integer_005"
            , "integer_006", "integer_007", "integer_008", "integer_009", "integer_010"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "time_001", "time_002"
            , "json_001", "json_002"
        ]
        sortPercepfuncArr = [
            "isckeck", "datacount",
            "isnotnull", "isnotnullper", "count", "discount",
            "sum", "avg", "max", "min", "round",
        ]

        pivot_table = pandas.pivot_table(df, values='log10diff', index=['columnname', 'percepfunc'],columns=['YYYY', 'MM', 'DD'], aggfunc='sum')
        pivot_table = pivot_table.sort_values(by='percepfunc', kind='mergesort', key=lambda x: pandas.Series(x).map(dict(zip(sortPercepfuncArr, range(len(sortPercepfuncArr))))))
        pivot_table = pivot_table.sort_values(by='columnname', kind='mergesort', key=lambda x: pandas.Series(x).map(dict(zip(sortColumnnameArr, range(len(sortColumnnameArr))))))

        plt.rcParams.update({'font.size': 6})

        plt.rcParams['font.sans-serif'] = ['DFKai-SB']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(19, 9.5))
        gs = GridSpec(1, 1)
        cm = plt.get_cmap("tab20c")
        ax = fig.add_subplot(gs[0:1, 0:1])

        # 添加標籤和顏色條
        sns.heatmap(pivot_table, cmap='RdBu', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False, vmin=-0.2,vmax=0.2)
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('left')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        ax.set_title("Data Perception {} {} {} {} {} ".format(fvInfo['RealTableName'], fvInfo['Product'], fvInfo['Project'],fvInfo['Tablename'], fvInfo['PercepCycle']))
        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry("+10+10")
        plt.tight_layout()
        pltPath = "{}/{}/file/DP".format(fvInfo['Product'], fvInfo['Project'])
        pltFile = "CPByDT_{}_{}_{}_{}_{}.png".format(fvInfo['RealTableName'], fvInfo['Product'], fvInfo['Project'], fvInfo['Tablename'], fvInfo['PercepCycle'] )
        os.makedirs(pltPath) if not os.path.isdir(pltPath) else None
        plt.savefig("{}/{}".format(pltPath,pltFile))

        return []

    @classmethod
    def compareDataPercoptionByTableName(self, fvInfo, otherInfo):
        import os
        import pandas
        import numpy as np
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        import Config
        
        pandas.set_option('display.float_format', '{:,.2f}'.format)
        
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        
        searchInitSQL = """
        select 
            AA.datatable
            , AA.product
            , AA.project
            , AA.datetime
            , AA.version
            , AA.columnname
            , AA.columnvalue
            , AA.percepfunc
            , AA.percepcycle
            , AA.percepvalue
            , AA.percepstate
            , AA.percepmemojson 
            , (AA.percepmemojson::json->>'diff')::DOUBLE PRECISION as diff
            , (AA.percepmemojson::json->>'log10diff')::DOUBLE PRECISION as log10diff
        from dataperception.dpmain AA
        WHERE 1 = 1 
            and AA.datatable = '[:Real_TABLE_NAME]'
            and AA.Product = '[:PRODUCT]'
            and AA.Project = '[:PROJECT]'
            and AA.version in ('[:MAINTABLENAME]','[:COMPARETABLENAME]')
            and AA.datetime = '[:DateNoLine]'
            AND AA.percepfunc not in ('datacount','isnotnull','isnotnullper','count', 'discount','sum')
            ; 
        """
        
        searchSQL = searchInitSQL.replace("[:Real_TABLE_NAME]", fvInfo["RealTableName"])
        searchSQL = searchSQL.replace("[:PRODUCT]", fvInfo["Product"])
        searchSQL = searchSQL.replace("[:PROJECT]", fvInfo["Project"])
        searchSQL = searchSQL.replace("[:MAINTABLENAME]", fvInfo["MainTablename"])
        searchSQL = searchSQL.replace("[:COMPARETABLENAME]", "','".join(fvInfo["CompareTablenames"]))
        searchSQL = searchSQL.replace("[:DateNoLine]", fvInfo["DataTime"].replace("-", ""))
        searchSQL = searchSQL.replace("[:PERCEP_CYCLE]", fvInfo["PercepCycle"])
        
        df = postgresCtrl.searchSQL(searchSQL)
        
        sortColumnnameArr = [
            "all_data"
            , "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "string_001", "string_002", "string_003", "string_004", "string_005"
            , "string_006", "string_007", "string_008", "string_009", "string_010"
            , "integer_001", "integer_002", "integer_003", "integer_004", "integer_005"
            , "integer_006", "integer_007", "integer_008", "integer_009", "integer_010"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "time_001", "time_002"
            , "json_001", "json_002"
        ]
        sortPercepfuncArr = [
            "isckeck", "datacount",
            "isnotnull", "isnotnullper", "count", "discount",
            "sum", "avg", "max", "min", "round",
        ]
        
        
        pivot_table = pandas.pivot_table(df, values='percepvalue', index=['columnname','percepfunc'], columns=['version'], aggfunc='sum' )
        pivot_table = pivot_table.sort_values(by='percepfunc', kind='mergesort', key=lambda x: pandas.Series(x).map(dict(zip(sortPercepfuncArr, range(len(sortPercepfuncArr))))))
        pivot_table = pivot_table.sort_values(by='columnname', kind='mergesort', key=lambda x: pandas.Series(x).map(dict(zip(sortColumnnameArr, range(len(sortColumnnameArr))))))
        
        for compareTablename in fvInfo["CompareTablenames"] :
            pivot_table[compareTablename] = np.log10(pivot_table[compareTablename]/ pivot_table[fvInfo["MainTablename"]])
        
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import seaborn as sns
        
        plt.rcParams.update({'font.size': 8})
        plt.rcParams['font.sans-serif'] = ['DFKai-SB']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(19, 9.5))
        gs = GridSpec(1, 1)
        cm = plt.get_cmap("tab20c")
        ax = fig.add_subplot(gs[0:1, 0:1])
        
        # 添加標籤和顏色條
        heatmap = sns.heatmap(pivot_table, cmap='RdBu', annot=True, fmt='.2f', linewidths=.5, ax=ax,cbar=False, vmin=-1, vmax=1)
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('left')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        ax.set_title("Data Perception {} {} {} {} {} ".format(fvInfo['RealTableName'],fvInfo['Product'],fvInfo['Project'],fvInfo['MainTablename'],fvInfo['PercepCycle']))
        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry("+10+10")
        plt.tight_layout()

        plt.savefig("Example/P81DataPerception/file/DP/CPByTable_{}_{}_{}_{}_{}.png".format(fvInfo['RealTableName'], fvInfo['Product'], fvInfo['Project'], fvInfo['MainTablename'], fvInfo['PercepCycle']))

        return []


