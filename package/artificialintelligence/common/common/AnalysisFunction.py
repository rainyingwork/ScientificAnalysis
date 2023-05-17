import os
class AnalysisFunction():

    def __init__(self):
        pass

    @classmethod
    def insertOverwriteAnalysisData(self,product,project,version,dt,analysisDataDF, useType= 'SQL') :
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        deleteSQL = """
             DELETE FROM observationdata.analysisdata
             WHERE 1 = 1 
                 AND product = '[:Product]'
                 AND project = '[:Project]'
                 AND version = '[:Version]'
                 AND dt = '[:DT]'
        """.replace("[:Product]", product) \
            .replace("[:Project]", project) \
            .replace("[:Version]", version) \
            .replace("[:DT]", dt)
        postgresCtrl.executeSQL(deleteSQL)
        self.insertAnalysisData(product, project, version, dt, analysisDataDF , useType)

    @classmethod

    def insertAnalysisData(self, product, project, version, dt, analysisDataDF , overWrite=False ,  useType='IO'):
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl

        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )

        analysisDataDF['product'] = product
        analysisDataDF['project'] = project
        analysisDataDF['version'] = version
        analysisDataDF['dt'] = dt

        tableFullName = "observationdata.analysisdata"

        deleteSQL = """
            DELETE FROM observationdata.analysisdata 
            WHERE 1 = 1 
                AND product = '[:Product]' 
                AND project = '[:Project]' 
                AND version = '[:Version]' 
                AND dt = '[:DateNoLine]' 
        """ .replace("[:Product]", product) \
            .replace("[:Project]", project) \
            .replace("[:Version]", version) \
            .replace("[:DateNoLine]", dt)

        postgresCtrl.executeSQL(deleteSQL)

        for column in AnalysisFunction.getAnalysisColumnNameArr():
            if column not in analysisDataDF.columns:
                analysisDataDF[column] = None
        analysisDataDF = analysisDataDF[self.getAnalysisColumnNameArr()]
        insertTableInfoDF = postgresCtrl.getTableInfoDF(tableFullName)
        if useType == 'SQL' :
            postgresCtrl.insertDataList(tableFullName, insertTableInfoDF, analysisDataDF)
        elif useType == 'IO' :
            postgresCtrl.insertDataByIO(tableFullName, insertTableInfoDF, analysisDataDF)

    @classmethod
    def getAnalysisColumnNameArr(self):
        columnNameArr = [
            "product", "project", "tablename", "dt"
            , "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "common_011", "common_012", "common_013", "common_014", "common_015"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "double_011", "double_012", "double_013", "double_014", "double_015"
            , "double_016", "double_017", "double_018", "double_019", "double_020"
            , "double_021", "double_022", "double_023", "double_024", "double_025"
            , "double_026", "double_027", "double_028", "double_029", "double_030"
            , "double_031", "double_032", "double_033", "double_034", "double_035"
            , "double_036", "double_037", "double_038", "double_039", "double_040"
            , "double_041", "double_042", "double_043", "double_044", "double_045"
            , "double_046", "double_047", "double_048", "double_049", "double_050"
            , "double_051", "double_052", "double_053", "double_054", "double_055"
            , "double_056", "double_057", "double_058", "double_059", "double_060"
            , "double_061", "double_062", "double_063", "double_064", "double_065"
            , "double_066", "double_067", "double_068", "double_069", "double_070"
            , "double_071", "double_072", "double_073", "double_074", "double_075"
            , "double_076", "double_077", "double_078", "double_079", "double_080"
            , "double_081", "double_082", "double_083", "double_084", "double_085"
            , "double_086", "double_087", "double_088", "double_089", "double_090"
            , "double_091", "double_092", "double_093", "double_094", "double_095"
            , "double_096", "double_097", "double_098", "double_099", "double_100"
            , "double_101", "double_102", "double_103", "double_104", "double_105"
            , "double_106", "double_107", "double_108", "double_109", "double_110"
            , "double_111", "double_112", "double_113", "double_114", "double_115"
            , "double_116", "double_117", "double_118", "double_119", "double_120"
            , "double_121", "double_122", "double_123", "double_124", "double_125"
            , "double_126", "double_127", "double_128", "double_129", "double_130"
            , "double_131", "double_132", "double_133", "double_134", "double_135"
            , "double_136", "double_137", "double_138", "double_139", "double_140"
            , "double_141", "double_142", "double_143", "double_144", "double_145"
            , "double_146", "double_147", "double_148", "double_149", "double_150"
            , "double_151", "double_152", "double_153", "double_154", "double_155"
            , "double_156", "double_157", "double_158", "double_159", "double_160"
            , "double_161", "double_162", "double_163", "double_164", "double_165"
            , "double_166", "double_167", "double_168", "double_169", "double_170"
            , "double_171", "double_172", "double_173", "double_174", "double_175"
            , "double_176", "double_177", "double_178", "double_179", "double_180"
            , "double_181", "double_182", "double_183", "double_184", "double_185"
            , "double_186", "double_187", "double_188", "double_189", "double_190"
            , "double_191", "double_192", "double_193", "double_194", "double_195"
            , "double_196", "double_197", "double_198", "double_199", "double_200"
            , "json_001"]
        return columnNameArr

    @classmethod
    def getDataColumnNameArr(self):
        columnNameArr = [
            "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "common_011", "common_012", "common_013", "common_014", "common_015"
            , "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "double_011", "double_012", "double_013", "double_014", "double_015"
            , "double_016", "double_017", "double_018", "double_019", "double_020"
            , "double_021", "double_022", "double_023", "double_024", "double_025"
            , "double_026", "double_027", "double_028", "double_029", "double_030"
            , "double_031", "double_032", "double_033", "double_034", "double_035"
            , "double_036", "double_037", "double_038", "double_039", "double_040"
            , "double_041", "double_042", "double_043", "double_044", "double_045"
            , "double_046", "double_047", "double_048", "double_049", "double_050"
            , "double_051", "double_052", "double_053", "double_054", "double_055"
            , "double_056", "double_057", "double_058", "double_059", "double_060"
            , "double_061", "double_062", "double_063", "double_064", "double_065"
            , "double_066", "double_067", "double_068", "double_069", "double_070"
            , "double_071", "double_072", "double_073", "double_074", "double_075"
            , "double_076", "double_077", "double_078", "double_079", "double_080"
            , "double_081", "double_082", "double_083", "double_084", "double_085"
            , "double_086", "double_087", "double_088", "double_089", "double_090"
            , "double_091", "double_092", "double_093", "double_094", "double_095"
            , "double_096", "double_097", "double_098", "double_099", "double_100"
            , "double_101", "double_102", "double_103", "double_104", "double_105"
            , "double_106", "double_107", "double_108", "double_109", "double_110"
            , "double_111", "double_112", "double_113", "double_114", "double_115"
            , "double_116", "double_117", "double_118", "double_119", "double_120"
            , "double_121", "double_122", "double_123", "double_124", "double_125"
            , "double_126", "double_127", "double_128", "double_129", "double_130"
            , "double_131", "double_132", "double_133", "double_134", "double_135"
            , "double_136", "double_137", "double_138", "double_139", "double_140"
            , "double_141", "double_142", "double_143", "double_144", "double_145"
            , "double_146", "double_147", "double_148", "double_149", "double_150"
            , "double_151", "double_152", "double_153", "double_154", "double_155"
            , "double_156", "double_157", "double_158", "double_159", "double_160"
            , "double_161", "double_162", "double_163", "double_164", "double_165"
            , "double_166", "double_167", "double_168", "double_169", "double_170"
            , "double_171", "double_172", "double_173", "double_174", "double_175"
            , "double_176", "double_177", "double_178", "double_179", "double_180"
            , "double_181", "double_182", "double_183", "double_184", "double_185"
            , "double_186", "double_187", "double_188", "double_189", "double_190"
            , "double_191", "double_192", "double_193", "double_194", "double_195"
            , "double_196", "double_197", "double_198", "double_199", "double_200"
            , "json_001"]
        return columnNameArr


    @classmethod
    def getCommonNameArr(self):
        columnNameArr = [
            "common_001", "common_002", "common_003", "common_004", "common_005"
            , "common_006", "common_007", "common_008", "common_009", "common_010"
            , "common_011", "common_012", "common_013", "common_014", "common_015"
        ]
        return columnNameArr

    @classmethod
    def getDoubleColumnArr(self):
        doubleColumnArr = [
            "double_001", "double_002", "double_003", "double_004", "double_005"
            , "double_006", "double_007", "double_008", "double_009", "double_010"
            , "double_011", "double_012", "double_013", "double_014", "double_015"
            , "double_016", "double_017", "double_018", "double_019", "double_020"
            , "double_021", "double_022", "double_023", "double_024", "double_025"
            , "double_026", "double_027", "double_028", "double_029", "double_030"
            , "double_031", "double_032", "double_033", "double_034", "double_035"
            , "double_036", "double_037", "double_038", "double_039", "double_040"
            , "double_041", "double_042", "double_043", "double_044", "double_045"
            , "double_046", "double_047", "double_048", "double_049", "double_050"
            , "double_051", "double_052", "double_053", "double_054", "double_055"
            , "double_056", "double_057", "double_058", "double_059", "double_060"
            , "double_061", "double_062", "double_063", "double_064", "double_065"
            , "double_066", "double_067", "double_068", "double_069", "double_070"
            , "double_071", "double_072", "double_073", "double_074", "double_075"
            , "double_076", "double_077", "double_078", "double_079", "double_080"
            , "double_081", "double_082", "double_083", "double_084", "double_085"
            , "double_086", "double_087", "double_088", "double_089", "double_090"
            , "double_091", "double_092", "double_093", "double_094", "double_095"
            , "double_096", "double_097", "double_098", "double_099", "double_100"
            , "double_101", "double_102", "double_103", "double_104", "double_105"
            , "double_106", "double_107", "double_108", "double_109", "double_110"
            , "double_111", "double_112", "double_113", "double_114", "double_115"
            , "double_116", "double_117", "double_118", "double_119", "double_120"
            , "double_121", "double_122", "double_123", "double_124", "double_125"
            , "double_126", "double_127", "double_128", "double_129", "double_130"
            , "double_131", "double_132", "double_133", "double_134", "double_135"
            , "double_136", "double_137", "double_138", "double_139", "double_140"
            , "double_141", "double_142", "double_143", "double_144", "double_145"
            , "double_146", "double_147", "double_148", "double_149", "double_150"
            , "double_151", "double_152", "double_153", "double_154", "double_155"
            , "double_156", "double_157", "double_158", "double_159", "double_160"
            , "double_161", "double_162", "double_163", "double_164", "double_165"
            , "double_166", "double_167", "double_168", "double_169", "double_170"
            , "double_171", "double_172", "double_173", "double_174", "double_175"
            , "double_176", "double_177", "double_178", "double_179", "double_180"
            , "double_181", "double_182", "double_183", "double_184", "double_185"
            , "double_186", "double_187", "double_188", "double_189", "double_190"
            , "double_191", "double_192", "double_193", "double_194", "double_195"
            , "double_196", "double_197", "double_198", "double_199", "double_200"]
        return doubleColumnArr

    @classmethod
    def getJsonColumnArr(self):
        JsonColumnArr = ["json_001"]
        return JsonColumnArr

    @classmethod
    def getAnalysisColumnDocInfo(self):
        # 本段隨然可以簡短，但此代碼未來會很長複製，請勿特別簡短
        columnInfoMap = {}
        columnInfoMap["product"] = {"description": "product" , 'datatype': 'string'}
        columnInfoMap["project"] = {"description": "project", 'datatype': 'string'}
        columnInfoMap["tablename"] = {"description": "tablename", 'datatype': 'string'}
        columnInfoMap["dt"] = {"description": "dt", 'datatype': 'string'}
        columnInfoMap["common_001"] = {"description": "common_001", 'datatype': 'string'}
        columnInfoMap["common_002"] = {"description": "common_002", 'datatype': 'string'}
        columnInfoMap["common_003"] = {"description": "common_003", 'datatype': 'string'}
        columnInfoMap["common_004"] = {"description": "common_004", 'datatype': 'string'}
        columnInfoMap["common_005"] = {"description": "common_005", 'datatype': 'string'}
        columnInfoMap["common_006"] = {"description": "common_006", 'datatype': 'string'}
        columnInfoMap["common_007"] = {"description": "common_007", 'datatype': 'string'}
        columnInfoMap["common_008"] = {"description": "common_008", 'datatype': 'string'}
        columnInfoMap["common_009"] = {"description": "common_009", 'datatype': 'string'}
        columnInfoMap["common_010"] = {"description": "common_010", 'datatype': 'string'}
        columnInfoMap["common_011"] = {"description": "common_011", 'datatype': 'string'}
        columnInfoMap["common_012"] = {"description": "common_012", 'datatype': 'string'}
        columnInfoMap["common_013"] = {"description": "common_013", 'datatype': 'string'}
        columnInfoMap["common_014"] = {"description": "common_014", 'datatype': 'string'}
        columnInfoMap["common_015"] = {"description": "common_015", 'datatype': 'string'}
        columnInfoMap["double_001"] = {"description": "double_001", 'datatype': 'double'}
        columnInfoMap["double_002"] = {"description": "double_002", 'datatype': 'double'}
        columnInfoMap["double_003"] = {"description": "double_003", 'datatype': 'double'}
        columnInfoMap["double_004"] = {"description": "double_004", 'datatype': 'double'}
        columnInfoMap["double_005"] = {"description": "double_005", 'datatype': 'double'}
        columnInfoMap["double_006"] = {"description": "double_006", 'datatype': 'double'}
        columnInfoMap["double_007"] = {"description": "double_007", 'datatype': 'double'}
        columnInfoMap["double_008"] = {"description": "double_008", 'datatype': 'double'}
        columnInfoMap["double_009"] = {"description": "double_009", 'datatype': 'double'}
        columnInfoMap["double_010"] = {"description": "double_010", 'datatype': 'double'}
        columnInfoMap["double_011"] = {"description": "double_011", 'datatype': 'double'}
        columnInfoMap["double_012"] = {"description": "double_012", 'datatype': 'double'}
        columnInfoMap["double_013"] = {"description": "double_013", 'datatype': 'double'}
        columnInfoMap["double_014"] = {"description": "double_014", 'datatype': 'double'}
        columnInfoMap["double_015"] = {"description": "double_015", 'datatype': 'double'}
        columnInfoMap["double_016"] = {"description": "double_016", 'datatype': 'double'}
        columnInfoMap["double_017"] = {"description": "double_017", 'datatype': 'double'}
        columnInfoMap["double_018"] = {"description": "double_018", 'datatype': 'double'}
        columnInfoMap["double_019"] = {"description": "double_019", 'datatype': 'double'}
        columnInfoMap["double_020"] = {"description": "double_020", 'datatype': 'double'}
        columnInfoMap["double_021"] = {"description": "double_021", 'datatype': 'double'}
        columnInfoMap["double_022"] = {"description": "double_022", 'datatype': 'double'}
        columnInfoMap["double_023"] = {"description": "double_023", 'datatype': 'double'}
        columnInfoMap["double_024"] = {"description": "double_024", 'datatype': 'double'}
        columnInfoMap["double_025"] = {"description": "double_025", 'datatype': 'double'}
        columnInfoMap["double_026"] = {"description": "double_026", 'datatype': 'double'}
        columnInfoMap["double_027"] = {"description": "double_027", 'datatype': 'double'}
        columnInfoMap["double_028"] = {"description": "double_028", 'datatype': 'double'}
        columnInfoMap["double_029"] = {"description": "double_029", 'datatype': 'double'}
        columnInfoMap["double_030"] = {"description": "double_030", 'datatype': 'double'}
        columnInfoMap["double_031"] = {"description": "double_031", 'datatype': 'double'}
        columnInfoMap["double_032"] = {"description": "double_032", 'datatype': 'double'}
        columnInfoMap["double_033"] = {"description": "double_033", 'datatype': 'double'}
        columnInfoMap["double_034"] = {"description": "double_034", 'datatype': 'double'}
        columnInfoMap["double_035"] = {"description": "double_035", 'datatype': 'double'}
        columnInfoMap["double_036"] = {"description": "double_036", 'datatype': 'double'}
        columnInfoMap["double_037"] = {"description": "double_037", 'datatype': 'double'}
        columnInfoMap["double_038"] = {"description": "double_038", 'datatype': 'double'}
        columnInfoMap["double_039"] = {"description": "double_039", 'datatype': 'double'}
        columnInfoMap["double_040"] = {"description": "double_040", 'datatype': 'double'}
        columnInfoMap["double_041"] = {"description": "double_041", 'datatype': 'double'}
        columnInfoMap["double_042"] = {"description": "double_042", 'datatype': 'double'}
        columnInfoMap["double_043"] = {"description": "double_043", 'datatype': 'double'}
        columnInfoMap["double_044"] = {"description": "double_044", 'datatype': 'double'}
        columnInfoMap["double_045"] = {"description": "double_045", 'datatype': 'double'}
        columnInfoMap["double_046"] = {"description": "double_046", 'datatype': 'double'}
        columnInfoMap["double_047"] = {"description": "double_047", 'datatype': 'double'}
        columnInfoMap["double_048"] = {"description": "double_048", 'datatype': 'double'}
        columnInfoMap["double_049"] = {"description": "double_049", 'datatype': 'double'}
        columnInfoMap["double_050"] = {"description": "double_050", 'datatype': 'double'}
        columnInfoMap["double_051"] = {"description": "double_051", 'datatype': 'double'}
        columnInfoMap["double_052"] = {"description": "double_052", 'datatype': 'double'}
        columnInfoMap["double_053"] = {"description": "double_053", 'datatype': 'double'}
        columnInfoMap["double_054"] = {"description": "double_054", 'datatype': 'double'}
        columnInfoMap["double_055"] = {"description": "double_055", 'datatype': 'double'}
        columnInfoMap["double_056"] = {"description": "double_056", 'datatype': 'double'}
        columnInfoMap["double_057"] = {"description": "double_057", 'datatype': 'double'}
        columnInfoMap["double_058"] = {"description": "double_058", 'datatype': 'double'}
        columnInfoMap["double_059"] = {"description": "double_059", 'datatype': 'double'}
        columnInfoMap["double_060"] = {"description": "double_060", 'datatype': 'double'}
        columnInfoMap["double_061"] = {"description": "double_061", 'datatype': 'double'}
        columnInfoMap["double_062"] = {"description": "double_062", 'datatype': 'double'}
        columnInfoMap["double_063"] = {"description": "double_063", 'datatype': 'double'}
        columnInfoMap["double_064"] = {"description": "double_064", 'datatype': 'double'}
        columnInfoMap["double_065"] = {"description": "double_065", 'datatype': 'double'}
        columnInfoMap["double_066"] = {"description": "double_066", 'datatype': 'double'}
        columnInfoMap["double_067"] = {"description": "double_067", 'datatype': 'double'}
        columnInfoMap["double_068"] = {"description": "double_068", 'datatype': 'double'}
        columnInfoMap["double_069"] = {"description": "double_069", 'datatype': 'double'}
        columnInfoMap["double_070"] = {"description": "double_070", 'datatype': 'double'}
        columnInfoMap["double_071"] = {"description": "double_071", 'datatype': 'double'}
        columnInfoMap["double_072"] = {"description": "double_072", 'datatype': 'double'}
        columnInfoMap["double_073"] = {"description": "double_073", 'datatype': 'double'}
        columnInfoMap["double_074"] = {"description": "double_074", 'datatype': 'double'}
        columnInfoMap["double_075"] = {"description": "double_075", 'datatype': 'double'}
        columnInfoMap["double_076"] = {"description": "double_076", 'datatype': 'double'}
        columnInfoMap["double_077"] = {"description": "double_077", 'datatype': 'double'}
        columnInfoMap["double_078"] = {"description": "double_078", 'datatype': 'double'}
        columnInfoMap["double_079"] = {"description": "double_079", 'datatype': 'double'}
        columnInfoMap["double_080"] = {"description": "double_080", 'datatype': 'double'}
        columnInfoMap["double_081"] = {"description": "double_081", 'datatype': 'double'}
        columnInfoMap["double_082"] = {"description": "double_082", 'datatype': 'double'}
        columnInfoMap["double_083"] = {"description": "double_083", 'datatype': 'double'}
        columnInfoMap["double_084"] = {"description": "double_084", 'datatype': 'double'}
        columnInfoMap["double_085"] = {"description": "double_085", 'datatype': 'double'}
        columnInfoMap["double_086"] = {"description": "double_086", 'datatype': 'double'}
        columnInfoMap["double_087"] = {"description": "double_087", 'datatype': 'double'}
        columnInfoMap["double_088"] = {"description": "double_088", 'datatype': 'double'}
        columnInfoMap["double_089"] = {"description": "double_089", 'datatype': 'double'}
        columnInfoMap["double_090"] = {"description": "double_090", 'datatype': 'double'}
        columnInfoMap["double_091"] = {"description": "double_091", 'datatype': 'double'}
        columnInfoMap["double_092"] = {"description": "double_092", 'datatype': 'double'}
        columnInfoMap["double_093"] = {"description": "double_093", 'datatype': 'double'}
        columnInfoMap["double_094"] = {"description": "double_094", 'datatype': 'double'}
        columnInfoMap["double_095"] = {"description": "double_095", 'datatype': 'double'}
        columnInfoMap["double_096"] = {"description": "double_096", 'datatype': 'double'}
        columnInfoMap["double_097"] = {"description": "double_097", 'datatype': 'double'}
        columnInfoMap["double_098"] = {"description": "double_098", 'datatype': 'double'}
        columnInfoMap["double_099"] = {"description": "double_099", 'datatype': 'double'}
        columnInfoMap["double_100"] = {"description": "double_100", 'datatype': 'double'}
        columnInfoMap["double_101"] = {"description": "double_101", 'datatype': 'double'}
        columnInfoMap["double_102"] = {"description": "double_102", 'datatype': 'double'}
        columnInfoMap["double_103"] = {"description": "double_103", 'datatype': 'double'}
        columnInfoMap["double_104"] = {"description": "double_104", 'datatype': 'double'}
        columnInfoMap["double_105"] = {"description": "double_105", 'datatype': 'double'}
        columnInfoMap["double_106"] = {"description": "double_106", 'datatype': 'double'}
        columnInfoMap["double_107"] = {"description": "double_107", 'datatype': 'double'}
        columnInfoMap["double_108"] = {"description": "double_108", 'datatype': 'double'}
        columnInfoMap["double_109"] = {"description": "double_109", 'datatype': 'double'}
        columnInfoMap["double_110"] = {"description": "double_110", 'datatype': 'double'}
        columnInfoMap["double_111"] = {"description": "double_111", 'datatype': 'double'}
        columnInfoMap["double_112"] = {"description": "double_112", 'datatype': 'double'}
        columnInfoMap["double_113"] = {"description": "double_113", 'datatype': 'double'}
        columnInfoMap["double_114"] = {"description": "double_114", 'datatype': 'double'}
        columnInfoMap["double_115"] = {"description": "double_115", 'datatype': 'double'}
        columnInfoMap["double_116"] = {"description": "double_116", 'datatype': 'double'}
        columnInfoMap["double_117"] = {"description": "double_117", 'datatype': 'double'}
        columnInfoMap["double_118"] = {"description": "double_118", 'datatype': 'double'}
        columnInfoMap["double_119"] = {"description": "double_119", 'datatype': 'double'}
        columnInfoMap["double_120"] = {"description": "double_120", 'datatype': 'double'}
        columnInfoMap["double_121"] = {"description": "double_121", 'datatype': 'double'}
        columnInfoMap["double_122"] = {"description": "double_122", 'datatype': 'double'}
        columnInfoMap["double_123"] = {"description": "double_123", 'datatype': 'double'}
        columnInfoMap["double_124"] = {"description": "double_124", 'datatype': 'double'}
        columnInfoMap["double_125"] = {"description": "double_125", 'datatype': 'double'}
        columnInfoMap["double_126"] = {"description": "double_126", 'datatype': 'double'}
        columnInfoMap["double_127"] = {"description": "double_127", 'datatype': 'double'}
        columnInfoMap["double_128"] = {"description": "double_128", 'datatype': 'double'}
        columnInfoMap["double_129"] = {"description": "double_129", 'datatype': 'double'}
        columnInfoMap["double_130"] = {"description": "double_130", 'datatype': 'double'}
        columnInfoMap["double_131"] = {"description": "double_131", 'datatype': 'double'}
        columnInfoMap["double_132"] = {"description": "double_132", 'datatype': 'double'}
        columnInfoMap["double_133"] = {"description": "double_133", 'datatype': 'double'}
        columnInfoMap["double_134"] = {"description": "double_134", 'datatype': 'double'}
        columnInfoMap["double_135"] = {"description": "double_135", 'datatype': 'double'}
        columnInfoMap["double_136"] = {"description": "double_136", 'datatype': 'double'}
        columnInfoMap["double_137"] = {"description": "double_137", 'datatype': 'double'}
        columnInfoMap["double_138"] = {"description": "double_138", 'datatype': 'double'}
        columnInfoMap["double_139"] = {"description": "double_139", 'datatype': 'double'}
        columnInfoMap["double_140"] = {"description": "double_140", 'datatype': 'double'}
        columnInfoMap["double_141"] = {"description": "double_141", 'datatype': 'double'}
        columnInfoMap["double_142"] = {"description": "double_142", 'datatype': 'double'}
        columnInfoMap["double_143"] = {"description": "double_143", 'datatype': 'double'}
        columnInfoMap["double_144"] = {"description": "double_144", 'datatype': 'double'}
        columnInfoMap["double_145"] = {"description": "double_145", 'datatype': 'double'}
        columnInfoMap["double_146"] = {"description": "double_146", 'datatype': 'double'}
        columnInfoMap["double_147"] = {"description": "double_147", 'datatype': 'double'}
        columnInfoMap["double_148"] = {"description": "double_148", 'datatype': 'double'}
        columnInfoMap["double_149"] = {"description": "double_149", 'datatype': 'double'}
        columnInfoMap["double_150"] = {"description": "double_150", 'datatype': 'double'}
        columnInfoMap["double_151"] = {"description": "double_151", 'datatype': 'double'}
        columnInfoMap["double_152"] = {"description": "double_152", 'datatype': 'double'}
        columnInfoMap["double_153"] = {"description": "double_153", 'datatype': 'double'}
        columnInfoMap["double_154"] = {"description": "double_154", 'datatype': 'double'}
        columnInfoMap["double_155"] = {"description": "double_155", 'datatype': 'double'}
        columnInfoMap["double_156"] = {"description": "double_156", 'datatype': 'double'}
        columnInfoMap["double_157"] = {"description": "double_157", 'datatype': 'double'}
        columnInfoMap["double_158"] = {"description": "double_158", 'datatype': 'double'}
        columnInfoMap["double_159"] = {"description": "double_159", 'datatype': 'double'}
        columnInfoMap["double_160"] = {"description": "double_160", 'datatype': 'double'}
        columnInfoMap["double_161"] = {"description": "double_161", 'datatype': 'double'}
        columnInfoMap["double_162"] = {"description": "double_162", 'datatype': 'double'}
        columnInfoMap["double_163"] = {"description": "double_163", 'datatype': 'double'}
        columnInfoMap["double_164"] = {"description": "double_164", 'datatype': 'double'}
        columnInfoMap["double_165"] = {"description": "double_165", 'datatype': 'double'}
        columnInfoMap["double_166"] = {"description": "double_166", 'datatype': 'double'}
        columnInfoMap["double_167"] = {"description": "double_167", 'datatype': 'double'}
        columnInfoMap["double_168"] = {"description": "double_168", 'datatype': 'double'}
        columnInfoMap["double_169"] = {"description": "double_169", 'datatype': 'double'}
        columnInfoMap["double_170"] = {"description": "double_170", 'datatype': 'double'}
        columnInfoMap["double_171"] = {"description": "double_171", 'datatype': 'double'}
        columnInfoMap["double_172"] = {"description": "double_172", 'datatype': 'double'}
        columnInfoMap["double_173"] = {"description": "double_173", 'datatype': 'double'}
        columnInfoMap["double_174"] = {"description": "double_174", 'datatype': 'double'}
        columnInfoMap["double_175"] = {"description": "double_175", 'datatype': 'double'}
        columnInfoMap["double_176"] = {"description": "double_176", 'datatype': 'double'}
        columnInfoMap["double_177"] = {"description": "double_177", 'datatype': 'double'}
        columnInfoMap["double_178"] = {"description": "double_178", 'datatype': 'double'}
        columnInfoMap["double_179"] = {"description": "double_179", 'datatype': 'double'}
        columnInfoMap["double_180"] = {"description": "double_180", 'datatype': 'double'}
        columnInfoMap["double_181"] = {"description": "double_181", 'datatype': 'double'}
        columnInfoMap["double_182"] = {"description": "double_182", 'datatype': 'double'}
        columnInfoMap["double_183"] = {"description": "double_183", 'datatype': 'double'}
        columnInfoMap["double_184"] = {"description": "double_184", 'datatype': 'double'}
        columnInfoMap["double_185"] = {"description": "double_185", 'datatype': 'double'}
        columnInfoMap["double_186"] = {"description": "double_186", 'datatype': 'double'}
        columnInfoMap["double_187"] = {"description": "double_187", 'datatype': 'double'}
        columnInfoMap["double_188"] = {"description": "double_188", 'datatype': 'double'}
        columnInfoMap["double_189"] = {"description": "double_189", 'datatype': 'double'}
        columnInfoMap["double_190"] = {"description": "double_190", 'datatype': 'double'}
        columnInfoMap["double_191"] = {"description": "double_191", 'datatype': 'double'}
        columnInfoMap["double_192"] = {"description": "double_192", 'datatype': 'double'}
        columnInfoMap["double_193"] = {"description": "double_193", 'datatype': 'double'}
        columnInfoMap["double_194"] = {"description": "double_194", 'datatype': 'double'}
        columnInfoMap["double_195"] = {"description": "double_195", 'datatype': 'double'}
        columnInfoMap["double_196"] = {"description": "double_196", 'datatype': 'double'}
        columnInfoMap["double_197"] = {"description": "double_197", 'datatype': 'double'}
        columnInfoMap["double_198"] = {"description": "double_198", 'datatype': 'double'}
        columnInfoMap["double_199"] = {"description": "double_199", 'datatype': 'double'}
        columnInfoMap["double_200"] = {"description": "double_200", 'datatype': 'double'}
        columnInfoMap["json_001"] = {"description": "json_001", 'datatype': 'string'}
        return columnInfoMap

    @classmethod
    def makeAnalysisColumnDocInfoByTagJson(self , columnInfoMap , tagText ) :
        for featureKey in tagText.getFeatureDict().keys():
            columnDict = tagText.getFeatureDict()[featureKey]
            columnInfoMap["double_{}".format(featureKey)] = {}
            columnInfoMap["double_{}".format(featureKey)]["description"] = columnDict['cnname']
            columnInfoMap["double_{}".format(featureKey)]["datatype"] = 'double'
            columnInfoMap["double_{}".format(featureKey)]["memo"] = columnDict['enname']
            commentMemo = "\n"
            commentMemo += "預處理方式：" + ",".join(columnDict['jsonmessage']['DataPreProcess']['ProcessingOrder'])
            columnInfoMap["double_{}".format(featureKey)]["commentMemo"] = commentMemo
        return columnInfoMap