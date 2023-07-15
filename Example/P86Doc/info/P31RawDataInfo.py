from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
from package.artificialintelligence.common.common.DocumentCtrl import DocumentCtrl as AnalysisDocumentCtrl
class RawDataInfo(RawDataFunction) :

    pass

    @classmethod
    def getInfo_R0_0_1(self, makeInfo):
        analysisDocumentCtrl = AnalysisDocumentCtrl()

        tableInfo = {
            "Project": "P86Doc"
            , "TableName": "R0_0_1"
            , "Memo": "P86Doc"
            , "DataType": "RawData"
            , "StartDate": "2023-01-01"
            , "EndDate": "2023-01-31"
            , "CommentMemo": "P86Doc CommentMemo"
        }

        checkFuncStrs = ["IsNotNull", "IsNotNullPer", "Count", "DisCount"]

        columnInfoMap = self.getAnalysisColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "專案", 'datatype': 'string'}
        columnInfoMap["project"] = {"description": "計畫", 'datatype': 'string'}
        columnInfoMap["version"] = {"description": "版本", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "日期", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "Id" , 'datatype': 'string' , "checkfuncs": checkFuncStrs}

        analysisDoubleInfoMap = analysisDocumentCtrl.makeAnalysisDoubleInfoByDataBase("Example", "P86Doc", "R0_0_0", makeInfo["DataTime"].replace("-", ""))
        for key in analysisDoubleInfoMap.keys():
            columnInfoMap[key] = analysisDoubleInfoMap[key]

        return columnInfoMap