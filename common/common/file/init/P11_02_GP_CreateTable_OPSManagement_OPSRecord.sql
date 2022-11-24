DROP TABLE IF EXISTS opsmanagement.opsrecord  ;

DROP sequence IF EXISTS opsmanagement.opsrecord_opsrecordid_seq ;

create sequence IF NOT EXISTS opsmanagement.opsrecord_opsrecordid_seq ;

CREATE TABLE IF NOT EXISTS opsmanagement.opsrecord (
	createtime timestamp NULL,
	modifytime timestamp NULL,
	deletetime timestamp NULL,
	opsrecordid INTEGER NOT NULL PRIMARY KEY default nextval('opsmanagement.opsrecord_opsrecordid_seq'),
	opsversion INTEGER NULL,
	opsorderjson text NULL,
	parameterjson text NULL,
	resultjson text null ,
	state text null
);

