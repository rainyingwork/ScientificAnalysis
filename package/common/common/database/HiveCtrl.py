from impala.dbapi import connect
from impala.util import as_pandas
import time

class HiveCtrl:
    # 初始化
    def __init__(self, host=None, port=None , user=None, password=None, database=None,auth_mechanism="PLAIN"):
        self.__host = host
        self.__port = port
        self.__user = user
        self.__password = password
        self.__database = database
        self.__authMechanism = auth_mechanism
        self.__reconnectsCount = 3

    # 執行SQL
    def executeSQL(self, sql):
        conn = connect (host=self.__host, port=self.__port , user=self.__user, password=self.__password, database=self.__database, auth_mechanism= self.__authMechanism)
        cursor = conn.cursor()
        excount = 1
        while 1 <= excount and excount <= self.__reconnectsCount:
            try:
                time.sleep((excount - 1) * 10)
                conn.close()
                return cursor.execute(sql)
            except:
                print(excount)
                excount = excount + 1
        print('info: Fail to execute SQL!')

    # 查詢SQL，回傳Dataframe
    def searchSQL(self, sql):
        conn = connect (host=self.__host, port=self.__port, user=self.__user, password=self.__password, database=self.__database, auth_mechanism=self.__authMechanism )
        cursor = conn.cursor()
        excount = 1
        while 1 <= excount and excount <= self.__reconnectsCount:
            try:
                time.sleep((excount - 1) * 10)
                cursor.execute(sql)
                df = as_pandas(cursor)
                conn.close()
                return df
            except:
                print(excount)
                excount = excount + 1
        print('info: Fail to search SQL!')


