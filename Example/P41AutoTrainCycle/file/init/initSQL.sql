CREATE TABLE originaldata.archive_countryinitialism (
	archiveid integer NULL,
	gender text NULL,
	age integer NULL,
	numberofkids integer NULL
);

CREATE TABLE originaldata.archive_pings (
	archiveid integer NULL,
	pingtime timestamp NULL
);

CREATE TABLE originaldata.archive_test (
	archiveid integer NULL,
	archivedate date NULL,
	onlinehours integer NULL
);