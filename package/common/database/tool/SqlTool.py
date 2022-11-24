
class SqlTool:
    # 载入SQL檔案
    def loadSQLfile(self, filePath, encoding="utf8"):
        self.__sqlfile = open(filePath, "r", encoding=encoding)

    # 將相關內容字串取代成相關字元
    def replaceSQLStrs(self,sqlReplaceArr=[["[:noReplace]",""]]):
        self.__sqlStrs = "".join(self.__sqlfile.readlines())
        for sqlReplace in sqlReplaceArr:
            self.__sqlStrs = self.__sqlStrs.replace(sqlReplace[0], sqlReplace[1])

    # 將字串切割成字串陣列
    def makeSQLArrSplitSQLStr(self):
        self.__sqlStrArr = self.__sqlStrs.split(";")[:-1]

    # 取得SQL字串陣列
    def getSQLArr(self):
        return self.__sqlStrArr
