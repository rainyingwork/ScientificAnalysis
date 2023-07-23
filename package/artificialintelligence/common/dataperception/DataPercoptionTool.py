class DataPercoptionTool():

    @staticmethod
    def makeBasicPercoptionSQL(fvInfo):
        basicPercoptionSQLs = """
      UPDATE dataperception.dptemp Set deletetime = now()
where 1 = 1		
    and datatable = '[:Real_TABLE_NAME]'
    and Product = '[:PRODUCT]'
    and Project = '[:PROJECT]'
    and version = '[:TABLENAME]'
    and datetime = '[:DateNoLine]' ;  

UPDATE dataperception.dpmain Set deletetime = now()
where 1 = 1
    and datatable = '[:Real_TABLE_NAME]'
    and Product = '[:PRODUCT]'
    and Project = '[:PROJECT]'
    and version = '[:TABLENAME]'
    and datetime = '[:DateNoLine]' ; 

-- 將感知結果變成寬表
insert into dataperception.dptemp 
with BASIC_DATE as (
[:BasicDataSQL]
) select 
    now() as createtime
    , now() as modifytime
    , null as deletetime 
    , nextval('dataperception.dataperception_dpmainid_seq') as dpmainid
    , realtablename as datatable
    , Product  as Product
    , Project as Project
    , dt as datetime 
    , tablename as version
    , SPLIT_PART((row).key, '_', 1) || '_' || SPLIT_PART((row).key, '_', 2) as columnname
    , columnvalue as columnvalue 
    , SPLIT_PART((row).key, '_', 3) as percepfunc
    , '[:PERCEP_CYCLE]' as percepcycle 
    , cast((row).value::text as double precision) as percepvalue 
    , 'CHECK_TEMP' as percepstate 
    , '{}' as percepmemojson 
from ([:PERCOP_SQL]) as dptemp ; 

-- 該筆代表該項目有執行檢查SQL
insert into dataperception.dptemp 
select 
    now() as createtime
    , now() as modifytime
    , null as deletetime 
    , nextval('dataperception.dataperception_dpmainid_seq') as dpmainid
    , '[:Real_TABLE_NAME]' as datatable
    , '[:PRODUCT]' as Product
    , '[:PROJECT]' as Project
    , '[:DateNoLine]' as datetime 
    , '[:TABLENAME]' as version
    , 'all_data' as columnname
    , null as columnvalue 
    , 'isckeck' as percepfunc
    , '[:PERCEP_CYCLE]' as percepcycle 
    , 1 as percepvalue 
    , 'CHECK_TEMP' as percepstate 
    , '{}' as percepmemojson ;

-- 感知現有資料
insert into dataperception.dpmain
select 
    AAAA.createtime
    , AAAA.modifytime	
    , AAAA.deletetime
    , AAAA.dptempid
    , AAAA.datatable
    , AAAA.product
    , AAAA.project
    , AAAA.datetime
    , AAAA.version
    , AAAA.columnname
    , AAAA.columnvalue
    , AAAA.percepfunc
    , AAAA.percepcycle
    , AAAA.percepvalue
    , 'CHECK_WAIT' as percepstate
    , to_jsonb(json_build_object('diff', AAAA.diff,'log10diff', AAAA.log10diff)) as percepmemojson 
from (
    select 
        AAA.*
        , case 
            when (AAA.percepvalue = 0 and BBB.percepvalue = 0) then 1
            when (BBB.percepvalue = 0 or BBB.percepvalue is null) then 9999999999
            when (AAA.percepvalue = 0 or AAA.percepvalue is null) then 0
            ELSE AAA.percepvalue/BBB.percepvalue
        end as diff
        , case 
            when (AAA.percepvalue = 0 and BBB.percepvalue = 0) then 0
            when (BBB.percepvalue = 0 or BBB.percepvalue is null) then 10
            when (AAA.percepvalue = 0 or AAA.percepvalue is null) then -10
            when (AAA.percepvalue/BBB.percepvalue < 0 and log10(AAA.percepvalue/BBB.percepvalue) > 0) then (log10(AAA.percepvalue/BBB.percepvalue) + 10)
            when (AAA.percepvalue/BBB.percepvalue < 0 and log10(AAA.percepvalue/BBB.percepvalue) < 0) then (log10(AAA.percepvalue/BBB.percepvalue) - 10)
            ELSE log10(AAA.percepvalue/BBB.percepvalue)
        end as log10diff
    from dataperception.dptemp AAA
    LEFT join (
        select 
            BB.columnname
            , BB.columnvalue
            , BB.percepfunc
            , SUM(BB.percepvalue)/7 as percepvalue
        from dataperception.dpmain BB
        where 1 = 1 
            and BB.datatable = '[:Real_TABLE_NAME]'
            and BB.product = '[:PRODUCT]'
            and BB.project = '[:PROJECT]'
            and case 
                    when '[:PERCEP_CYCLE]' = '1D' then BB.datetime <= to_char((date '[:DateNoLine]' + integer '-1'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1W' then BB.datetime <= to_char((date '[:DateNoLine]' + integer '-7'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1M' then BB.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '1 month') + interval '1 month' - interval '1 day','yyyyMMdd')
                end
            and case 
                    when '[:PERCEP_CYCLE]' = '1D' then BB.datetime >= to_char((date '[:DateNoLine]' + integer '-7'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1W' then BB.datetime >= to_char((date '[:DateNoLine]' + integer '-49'),'yyyyMMdd')
                    when '[:PERCEP_CYCLE]' = '1M' then BB.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '7 month') ,'yyyyMMdd')
                end
            and BB.version = '[:TABLENAME]'
            and BB.percepcycle = '[:PERCEP_CYCLE]'
            and BB.deletetime is null 
        group by 
            BB.columnname
            , BB.columnvalue
            , BB.percepfunc
    ) BBB on 1 = 1
        and BBB.columnname = AAA.columnname
        and ( 1 != 1 
            OR BBB.columnvalue = AAA.columnvalue
            OR ( BBB.columnvalue is null and AAA.columnvalue is null )
        ) 
        and BBB.percepfunc = AAA.percepfunc
    where 1 = 1 
        and AAA.datatable = '[:Real_TABLE_NAME]'
        and AAA.product = '[:PRODUCT]'
        and AAA.project = '[:PROJECT]'
        and AAA.datetime = '[:DateNoLine]'
        and AAA.version = '[:TABLENAME]'
        and AAA.percepcycle = '[:PERCEP_CYCLE]'
        and AAA.deletetime is null 
) AAAA ; 

-- 感知過往可能錯誤資料
insert into dataperception.dpmain
select 
    now() as createtime
    , now() as modifytime
    , null as deletetime 
    , nextval('dataperception.dataperception_dpmainid_seq') as dpmainid
    , '[:Real_TABLE_NAME]' as datatable
    , '[:PRODUCT]' as Product
    , '[:PROJECT]' as Project
    , '[:DateNoLine]' as datetime 
    , '[:TABLENAME]' as version
    , AAA.columnname
    , AAA.columnvalue
    , AAA.percepfunc
    , '[:PERCEP_CYCLE]' as percepcycle 
    , 0 as percepvalue 
    , 'CHECK_WAIT' as percepstate 
    , to_jsonb(json_build_object('diff', 0,'log10diff', -10)) as percepmemojson 
from dataperception.dpmain AAA
left join (
    select 
        BB.columnname
        , BB.columnvalue
        , BB.percepfunc
    from dataperception.dpmain BB
    where 1 = 1 
        and BB.datatable = '[:Real_TABLE_NAME]'
        and BB.product = '[:PRODUCT]'
        and BB.project = '[:PROJECT]'
        and BB.datetime = '[:DateNoLine]'
        and BB.version = '[:TABLENAME]'
        and BB.percepcycle = '[:PERCEP_CYCLE]'
        and BB.deletetime is null 	
) BBB on 1 = 1 
    and BBB.columnname = AAA.columnname
    and ( 1 != 1 
        OR BBB.columnvalue = AAA.columnvalue
        OR ( BBB.columnvalue is null and AAA.columnvalue is null )
    ) 
    and BBB.percepfunc = AAA.percepfunc
where 1 = 1 
    and AAA.datatable = '[:Real_TABLE_NAME]'
    and AAA.product = '[:PRODUCT]'
    and AAA.project = '[:PROJECT]'
    and case 
            when '[:PERCEP_CYCLE]' = '1D' then AAA.datetime <= to_char((date '[:DateNoLine]' + integer '-1'),'yyyyMMdd')
            when '[:PERCEP_CYCLE]' = '1W' then AAA.datetime <= to_char((date '[:DateNoLine]' + integer '-7'),'yyyyMMdd')
            when '[:PERCEP_CYCLE]' = '1M' then AAA.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '1 month') + interval '1 month' - interval '1 day','yyyyMMdd')
        end
    and case 
            when '[:PERCEP_CYCLE]' = '1D' then AAA.datetime >= to_char((date '[:DateNoLine]' + integer '-7'),'yyyyMMdd')
            when '[:PERCEP_CYCLE]' = '1W' then AAA.datetime >= to_char((date '[:DateNoLine]' + integer '-49'),'yyyyMMdd')
            when '[:PERCEP_CYCLE]' = '1M' then AAA.datetime <= to_char(date_trunc('month', date '[:DateNoLine]' - interval '7 month') ,'yyyyMMdd')
        end
    and AAA.version = '[:TABLENAME]'
    and AAA.percepcycle = '[:PERCEP_CYCLE]'
    and AAA.deletetime is null 
    and BBB.percepfunc is null 
group by 
    AAA.columnname
    , AAA.columnvalue
    , AAA.percepfunc ; 

UPDATE dataperception.dpmain Set percepstate = 'CHECK_FINISH'
where 1 = 1
    and Product = '[:PRODUCT]'
    and Project = '[:PROJECT]'
    and version = '[:TABLENAME]'
    and datetime = '[:DateNoLine]'
    and percepstate = 'CHECK_WAIT'
    and (percepmemojson::json->>'log10diff')::DOUBLE PRECISION <= 0.15 
    and (percepmemojson::json->>'log10diff')::DOUBLE PRECISION >= -0.15 ; 

UPDATE dataperception.dpmain Set percepstate = 'CHECK_ERROR'
where 1 = 1
    and Product = '[:PRODUCT]'
    and Project = '[:PROJECT]'
    and version = '[:TABLENAME]'
    and datetime = '[:DateNoLine]'
    and percepstate = 'CHECK_WAIT' ; 

DELETE FROM dataperception.dptemp 
where 1 = 1		
    and datatable = '[:Real_TABLE_NAME]'
    and Product = '[:PRODUCT]'
    and Project = '[:PROJECT]'
    and version = '[:TABLENAME]'
    and datetime = '[:DateNoLine]' ;    
"""
        if fvInfo["RealTableName"] == "observationdata.standarddata" :
            basicDataSQL = """
    SELECT
        '[:Real_TABLE_NAME]' as realtablename
        , Product
        , Project
        , tablename
        , dt
        , [:PERCOP_COLUMN_VALSE_SELECT] as columnvalue
        , SUM(1) AS all_data_datacount [:PERCOP_DETAIL]
    from [:Real_TABLE_NAME] AA
    where 1 = 1
        and Product = '[:PRODUCT]'
        and Project = '[:PROJECT]'
        and tablename = '[:TABLENAME]'
        and dt = '[:DateNoLine]'
    group by
        Product
        , Project
        , tablename
        , dt [:PERCOP_COLUMN_VALSE_GROUP]
"""
            return basicPercoptionSQLs.replace("[:BasicDataSQL]",basicDataSQL)
        elif fvInfo["RealTableName"] == "observationdata.analysisdata" :
            basicDataSQL = """
    SELECT
        '[:Real_TABLE_NAME]' as realtablename
        , Product
        , Project
        , version
        , dt
        , [:PERCOP_COLUMN_VALSE_SELECT] as columnvalue
        , SUM(1) AS all_data_datacount [:PERCOP_DETAIL]
    from [:Real_TABLE_NAME] AA
    where 1 = 1
        and Product = '[:PRODUCT]'
        and Project = '[:PROJECT]'
        and version = '[:TABLENAME]'
        and dt = '[:DateNoLine]'
    group by
        Product
        , Project
        , version
        , dt [:PERCOP_COLUMN_VALSE_GROUP]
"""
            return basicPercoptionSQLs.replace("[:BasicDataSQL]", basicDataSQL)

    @staticmethod
    def makeBasicPercoptionDetailSQL(fvInfo):
        if fvInfo["RealTableName"] == "observationdata.standarddata" :
            return """
            select 
                realtablename , Product , Project , tablename , dt , columnvalue
                , json_each(
                    json_build_object(
                        [:PERCOP_COLUMN]
                    )
                ) as row 
            FROM BASIC_DATE  
            """
        elif fvInfo["RealTableName"] == "observationdata.analysisdata" :
            return """
            select 
                realtablename , Product , Project , version , dt , columnvalue
                , json_each(
                    json_build_object(
                        [:PERCOP_COLUMN]
                    )
                ) as row 
            FROM BASIC_DATE  
            """

    @staticmethod
    def makePercoptionIsNotNullSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_isnotnull', \"{}_isnotnull\" ".format(columnname, columnname)
        if columnInfo['datatype'] == 'string':
            percopDetailSQL = "SUM( CASE WHEN AA.{} IS NULL THEN 0 WHEN AA.{} = {} THEN 0 ELSE 1 END) AS {}_isnotnull ".format(
                columnname, columnname, "''", columnname)
        else:
            percopDetailSQL = "SUM( CASE WHEN AA.{} IS NULL THEN 0 WHEN AA.{} = {} THEN 0 ELSE 1 END) AS {}_isnotnull ".format(
                columnname, columnname, "0", columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionIsNotNullPerSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_isnotnullper', \"{}_isnotnullper\" ".format(columnname, columnname)
        if columnInfo['datatype'] == 'string':
            percopDetailSQL = "SUM(CASE WHEN AA.{} IS NULL THEN 0 WHEN AA.{} = {} THEN 0 ELSE 1 END)/SUM(1) AS {}_isnotnullper ".format(
                columnname, columnname, "''", columnname)
        else:
            percopDetailSQL = "SUM(CASE WHEN AA.{} IS NULL THEN 0 WHEN AA.{} = {} THEN 0 ELSE 1 END)/SUM(1) AS {}_isnotnullper ".format(
                columnname, columnname, "0", columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionCountSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_count', \"{}_count\" ".format(columnname, columnname)
        percopDetailSQL = "count( AA.{}) AS {}_count ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionDisCountSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_discount', \"{}_discount\" ".format(columnname, columnname)
        percopDetailSQL = "count( distinct AA.{}) AS {}_discount ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionSumSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_sum', \"{}_sum\" ".format(columnname, columnname)
        percopDetailSQL = "sum(AA.{}) AS {}_sum ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionAvgSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_avg', \"{}_avg\" ".format(columnname, columnname)
        percopDetailSQL = "avg(AA.{}) AS {}_avg ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionMaxSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_max', \"{}_max\" ".format(columnname, columnname)
        percopDetailSQL = "Max(AA.{}) AS {}_max ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionMinSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_min', \"{}_min\" ".format(columnname, columnname)
        percopDetailSQL = "Min(AA.{}) AS {}_min ".format(columnname, columnname)
        return percopColumnSQL, percopDetailSQL

    @staticmethod
    def makePercoptionRoundSQL(columnInfo):
        columnname = columnInfo['columnname']
        percopColumnSQL = "'{}_round', \"{}_round\" ".format(columnname, columnname)
        percopDetailSQL = "Max(AA.{}) - Min(AA.{}) AS {}_round ".format(columnname, columnname, columnname)
        return percopColumnSQL, percopDetailSQL