DROP TABLE IF EXISTS observationdata.standarddata ;

CREATE TABLE IF NOT EXISTS observationdata.standarddata (
    product text
    , project text
    , tablename text
    , dt text
    , common_001 text
    , common_002 text
    , common_003 text
    , common_004 text
    , common_005 text
    , common_006 text
    , common_007 text
    , common_008 text
    , common_009 text
    , common_010 text
    , string_001 text
    , string_002 text
    , string_003 text
    , string_004 text
    , string_005 text
    , string_006 text
    , string_007 text
    , string_008 text
    , string_009 text
    , string_010 text
    , integer_001 bigint
    , integer_002 bigint
    , integer_003 bigint
    , integer_004 bigint
    , integer_005 bigint
    , integer_006 bigint
    , integer_007 bigint
    , integer_008 bigint
    , integer_009 bigint
    , integer_010 bigint
    , double_001 double precision
    , double_002 double precision
    , double_003 double precision
    , double_004 double precision
    , double_005 double precision
    , double_006 double precision
    , double_007 double precision
    , double_008 double precision
    , double_009 double precision
    , double_010 double precision
    , time_001 timestamp
    , time_002 timestamp
    , json_001 text
    , json_002 text
) --PARTITION BY RANGE (product , project ,tablename,  dt) ;

CREATE INDEX standarddata_index ON observationdata.standarddata (product,project,tablename,dt);
