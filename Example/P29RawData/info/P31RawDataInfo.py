from package.artificialintelligence.common.common.AnalysisFunction import AnalysisFunction
from package.artificialintelligence.common.virtualentity.TagText import TagText

class RawDataInfo(AnalysisFunction) :

    @classmethod
    def getInfo_R0_0_0(self, makeInfo=None):

        tableInfo = {
            "Project": "P29RawData"
            , "TableName": "S0_0_1"
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
        columnInfoMap["tablename"] = {"description": "R0_0_0", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "日期", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "Id", 'datatype': 'string'}

        tagText = TagText()
        tagText.setFeatureDictByFilePath("Example/P29RawData/file/TagText/TagR0_0_0.json")
        columnInfoMap = self.makeAnalysisColumnDocInfoByTagJson(columnInfoMap,tagText )

        return columnInfoMap


