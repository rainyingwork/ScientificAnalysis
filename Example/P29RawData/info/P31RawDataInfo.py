from package.artificialintelligence.common.common.AnalysisFunction import AnalysisFunction
from package.artificialintelligence.common.virtualentity.TagText import TagText

class RawDataInfo(AnalysisFunction) :

    @classmethod
    def getInfo_R0_0_1_ByFile(self, makeInfo=None):

        tableInfo = {
            "Project": "P29RawData"
            , "TableName": "R0_0_1"
            , "Memo": "P29RawData"
            , "DataType": "Analysis"
            , "StartDate": "2020-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getAnalysisColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "Example", 'datatype': 'string'}
        columnInfoMap["project"] = {"description": "P29RawData", 'datatype': 'string'}
        columnInfoMap["tablename"] = {"description": "R0_0_1", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "日期", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "Id", 'datatype': 'string'}

        tagText = TagText()
        tagText.setFeatureDictByFilePath("Example/P29RawData/file/TagText/TagR0_0_0.json")
        columnInfoMap = self.makeAnalysisColumnDocInfoByTagJson(columnInfoMap,tagText )

        return columnInfoMap

    @classmethod
    def getInfo_R0_0_1_ByDB(self, makeInfo=None):
        from package.artificialintelligence.common.common.DocumentCtrl import DocumentCtrl as AnalysisDocumentCtrl
        analysisDocumentCtrl = AnalysisDocumentCtrl()
        tableInfo = {
            "Project": "P29RawData"
            , "TableName": "R0_0_1"
            , "Memo": "P29RawData"
            , "DataType": "Analysis"
            , "StartDate": "2020-01-01"
            , "EndDate": "2050-12-31"
            , "CommentMemo": "P13Standard CommentMemo"
        }

        columnInfoMap = self.getAnalysisColumnDocInfo()
        columnInfoMap["TableInfo"] = tableInfo
        columnInfoMap["product"] = {"description": "Example", 'datatype': 'string'}
        columnInfoMap["project"] = {"description": "P29RawData", 'datatype': 'string'}
        columnInfoMap["tablename"] = {"description": "R0_0_1", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "日期", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "Id", 'datatype': 'string'}


        analysisDoubleInfoMap = analysisDocumentCtrl.makeAnalysisDoubleInfoByDataBase("Example", "P29RawData", "R0_0_0", "20220101")
        for key in analysisDoubleInfoMap.keys():
            columnInfoMap[key] = analysisDoubleInfoMap[key]

        return columnInfoMap
