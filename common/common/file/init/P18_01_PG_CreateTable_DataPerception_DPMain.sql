DROP TABLE IF EXISTS dataperception.dpmain  ;

DROP sequence IF EXISTS dataperception.dataperception_dpmainid_seq ;

create sequence IF NOT EXISTS dataperception.dataperception_dpmainid_seq ;

CREATE TABLE IF NOT EXISTS dataperception.dpmain (
	createtime timestamp NULL,
	modifytime timestamp NULL,
	deletetime timestamp NULL,
	dpmainid INTEGER NOT NULL PRIMARY KEY default nextval('dataperception.dataperception_dpmainid_seq'),
	datatable text NULL,
	product text NULL,
	project text NULL,
	datetime text NULL,
	version text NULL,
	columnname text NULL,
	columnvalue text NULL,
	percepfunc text NULL,
    percepcycle text NULL,
    percepvalue text NULL,
    percepstate text NULL,
	percepmemojson text NULL
) ;

