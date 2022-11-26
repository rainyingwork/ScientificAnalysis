DROP TABLE IF EXISTS opsmanagement.opsdetail  ;

DROP sequence IF EXISTS opsmanagement.opsdetail_opsdetailid_seq ;

create sequence IF NOT EXISTS opsmanagement.opsdetail_opsdetailid_seq ;

CREATE TABLE IF NOT EXISTS opsmanagement.opsdetail (
	createtime timestamp NULL,
	modifytime timestamp NULL,
	deletetime timestamp NULL,
	opsdetailid INTEGER NOT NULL PRIMARY KEY default nextval('opsmanagement.opsdetail_opsdetailid_seq'),
	opsrecord INTEGER NULL,
	exefunction text NULL,
	parameterjson text NULL,
	resultjson text null ,
	state text null
);

