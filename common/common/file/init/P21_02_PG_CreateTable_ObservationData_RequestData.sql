DROP TABLE IF EXISTS observationdata.requestdata ;

DROP sequence IF EXISTS observationdata.requestdata_requestdataid_seq ;

create sequence IF NOT EXISTS observationdata.requestdata_requestdataid_seq ;

CREATE TABLE IF NOT EXISTS observationdata.requestdata (
    createtime timestamp NULL
	, modifytime timestamp NULL
	, deletetime timestamp NULL
	, requestdataid INTEGER NOT NULL PRIMARY KEY default nextval('observationdata.requestdata_requestdataid_seq')
    , product text
    , project text
    , exefunction text
    , startdate text
    , enddate text
    , requesturl TEXT
    , requesttype TEXT
    , requestparameter TEXT
    , requesttitle TEXT
    , requestcontent TEXT
) --PARTITION BY RANGE (product , project ,tablename,  dt) ;
