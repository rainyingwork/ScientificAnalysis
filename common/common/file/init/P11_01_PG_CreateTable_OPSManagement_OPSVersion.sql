DROP TABLE IF EXISTS opsmanagement.opsversion  ;

DROP sequence IF EXISTS opsmanagement.opsversion_opsversionid_seq ;

create sequence IF NOT EXISTS opsmanagement.opsversion_opsversionid_seq ;

CREATE TABLE IF NOT EXISTS opsmanagement.opsversion (
	createtime timestamp NULL,
	modifytime timestamp NULL,
	deletetime timestamp NULL,
	opsversionid INTEGER NOT NULL PRIMARY KEY default nextval('opsmanagement.opsversion_opsversionid_seq'),
	product text NULL,
	project text NULL,
	opsversion text NULL,
	opsorderjson text NULL,
	parameterjson text NULL,
	resultjson text NULL
) ;


