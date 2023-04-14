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
        columnInfoMap["product"] = {"description": "product"}
        columnInfoMap["project"] = {"description": "project"}
        columnInfoMap["tablename"] = {"description": "tablename"}
        columnInfoMap["dt"] = {"description": "dt"}
        columnInfoMap["common_001"] = {"description": "common_001"}
        columnInfoMap["common_002"] = {"description": "common_002"}
        columnInfoMap["common_003"] = {"description": "common_003"}
        columnInfoMap["common_004"] = {"description": "common_004"}
        columnInfoMap["common_005"] = {"description": "common_005"}
        columnInfoMap["common_006"] = {"description": "common_006"}
        columnInfoMap["common_007"] = {"description": "common_007"}
        columnInfoMap["common_008"] = {"description": "common_008"}
        columnInfoMap["common_009"] = {"description": "common_009"}
        columnInfoMap["common_010"] = {"description": "common_010"}
        columnInfoMap["common_011"] = {"description": "common_011"}
        columnInfoMap["common_012"] = {"description": "common_012"}
        columnInfoMap["common_013"] = {"description": "common_013"}
        columnInfoMap["common_014"] = {"description": "common_014"}
        columnInfoMap["common_015"] = {"description": "common_015"}
        columnInfoMap["double_001"] = {"description": "double_001"}
        columnInfoMap["double_002"] = {"description": "double_002"}
        columnInfoMap["double_003"] = {"description": "double_003"}
        columnInfoMap["double_004"] = {"description": "double_004"}
        columnInfoMap["double_005"] = {"description": "double_005"}
        columnInfoMap["double_006"] = {"description": "double_006"}
        columnInfoMap["double_007"] = {"description": "double_007"}
        columnInfoMap["double_008"] = {"description": "double_008"}
        columnInfoMap["double_009"] = {"description": "double_009"}
        columnInfoMap["double_010"] = {"description": "double_010"}
        columnInfoMap["double_011"] = {"description": "double_011"}
        columnInfoMap["double_012"] = {"description": "double_012"}
        columnInfoMap["double_013"] = {"description": "double_013"}
        columnInfoMap["double_014"] = {"description": "double_014"}
        columnInfoMap["double_015"] = {"description": "double_015"}
        columnInfoMap["double_016"] = {"description": "double_016"}
        columnInfoMap["double_017"] = {"description": "double_017"}
        columnInfoMap["double_018"] = {"description": "double_018"}
        columnInfoMap["double_019"] = {"description": "double_019"}
        columnInfoMap["double_020"] = {"description": "double_020"}
        columnInfoMap["double_021"] = {"description": "double_021"}
        columnInfoMap["double_022"] = {"description": "double_022"}
        columnInfoMap["double_023"] = {"description": "double_023"}
        columnInfoMap["double_024"] = {"description": "double_024"}
        columnInfoMap["double_025"] = {"description": "double_025"}
        columnInfoMap["double_026"] = {"description": "double_026"}
        columnInfoMap["double_027"] = {"description": "double_027"}
        columnInfoMap["double_028"] = {"description": "double_028"}
        columnInfoMap["double_029"] = {"description": "double_029"}
        columnInfoMap["double_030"] = {"description": "double_030"}
        columnInfoMap["double_031"] = {"description": "double_031"}
        columnInfoMap["double_032"] = {"description": "double_032"}
        columnInfoMap["double_033"] = {"description": "double_033"}
        columnInfoMap["double_034"] = {"description": "double_034"}
        columnInfoMap["double_035"] = {"description": "double_035"}
        columnInfoMap["double_036"] = {"description": "double_036"}
        columnInfoMap["double_037"] = {"description": "double_037"}
        columnInfoMap["double_038"] = {"description": "double_038"}
        columnInfoMap["double_039"] = {"description": "double_039"}
        columnInfoMap["double_040"] = {"description": "double_040"}
        columnInfoMap["double_041"] = {"description": "double_041"}
        columnInfoMap["double_042"] = {"description": "double_042"}
        columnInfoMap["double_043"] = {"description": "double_043"}
        columnInfoMap["double_044"] = {"description": "double_044"}
        columnInfoMap["double_045"] = {"description": "double_045"}
        columnInfoMap["double_046"] = {"description": "double_046"}
        columnInfoMap["double_047"] = {"description": "double_047"}
        columnInfoMap["double_048"] = {"description": "double_048"}
        columnInfoMap["double_049"] = {"description": "double_049"}
        columnInfoMap["double_050"] = {"description": "double_050"}
        columnInfoMap["double_051"] = {"description": "double_051"}
        columnInfoMap["double_052"] = {"description": "double_052"}
        columnInfoMap["double_053"] = {"description": "double_053"}
        columnInfoMap["double_054"] = {"description": "double_054"}
        columnInfoMap["double_055"] = {"description": "double_055"}
        columnInfoMap["double_056"] = {"description": "double_056"}
        columnInfoMap["double_057"] = {"description": "double_057"}
        columnInfoMap["double_058"] = {"description": "double_058"}
        columnInfoMap["double_059"] = {"description": "double_059"}
        columnInfoMap["double_060"] = {"description": "double_060"}
        columnInfoMap["double_061"] = {"description": "double_061"}
        columnInfoMap["double_062"] = {"description": "double_062"}
        columnInfoMap["double_063"] = {"description": "double_063"}
        columnInfoMap["double_064"] = {"description": "double_064"}
        columnInfoMap["double_065"] = {"description": "double_065"}
        columnInfoMap["double_066"] = {"description": "double_066"}
        columnInfoMap["double_067"] = {"description": "double_067"}
        columnInfoMap["double_068"] = {"description": "double_068"}
        columnInfoMap["double_069"] = {"description": "double_069"}
        columnInfoMap["double_070"] = {"description": "double_070"}
        columnInfoMap["double_071"] = {"description": "double_071"}
        columnInfoMap["double_072"] = {"description": "double_072"}
        columnInfoMap["double_073"] = {"description": "double_073"}
        columnInfoMap["double_074"] = {"description": "double_074"}
        columnInfoMap["double_075"] = {"description": "double_075"}
        columnInfoMap["double_076"] = {"description": "double_076"}
        columnInfoMap["double_077"] = {"description": "double_077"}
        columnInfoMap["double_078"] = {"description": "double_078"}
        columnInfoMap["double_079"] = {"description": "double_079"}
        columnInfoMap["double_080"] = {"description": "double_080"}
        columnInfoMap["double_081"] = {"description": "double_081"}
        columnInfoMap["double_082"] = {"description": "double_082"}
        columnInfoMap["double_083"] = {"description": "double_083"}
        columnInfoMap["double_084"] = {"description": "double_084"}
        columnInfoMap["double_085"] = {"description": "double_085"}
        columnInfoMap["double_086"] = {"description": "double_086"}
        columnInfoMap["double_087"] = {"description": "double_087"}
        columnInfoMap["double_088"] = {"description": "double_088"}
        columnInfoMap["double_089"] = {"description": "double_089"}
        columnInfoMap["double_090"] = {"description": "double_090"}
        columnInfoMap["double_091"] = {"description": "double_091"}
        columnInfoMap["double_092"] = {"description": "double_092"}
        columnInfoMap["double_093"] = {"description": "double_093"}
        columnInfoMap["double_094"] = {"description": "double_094"}
        columnInfoMap["double_095"] = {"description": "double_095"}
        columnInfoMap["double_096"] = {"description": "double_096"}
        columnInfoMap["double_097"] = {"description": "double_097"}
        columnInfoMap["double_098"] = {"description": "double_098"}
        columnInfoMap["double_099"] = {"description": "double_099"}
        columnInfoMap["double_100"] = {"description": "double_100"}
        columnInfoMap["double_101"] = {"description": "double_101"}
        columnInfoMap["double_102"] = {"description": "double_102"}
        columnInfoMap["double_103"] = {"description": "double_103"}
        columnInfoMap["double_104"] = {"description": "double_104"}
        columnInfoMap["double_105"] = {"description": "double_105"}
        columnInfoMap["double_106"] = {"description": "double_106"}
        columnInfoMap["double_107"] = {"description": "double_107"}
        columnInfoMap["double_108"] = {"description": "double_108"}
        columnInfoMap["double_109"] = {"description": "double_109"}
        columnInfoMap["double_110"] = {"description": "double_110"}
        columnInfoMap["double_111"] = {"description": "double_111"}
        columnInfoMap["double_112"] = {"description": "double_112"}
        columnInfoMap["double_113"] = {"description": "double_113"}
        columnInfoMap["double_114"] = {"description": "double_114"}
        columnInfoMap["double_115"] = {"description": "double_115"}
        columnInfoMap["double_116"] = {"description": "double_116"}
        columnInfoMap["double_117"] = {"description": "double_117"}
        columnInfoMap["double_118"] = {"description": "double_118"}
        columnInfoMap["double_119"] = {"description": "double_119"}
        columnInfoMap["double_120"] = {"description": "double_120"}
        columnInfoMap["double_121"] = {"description": "double_121"}
        columnInfoMap["double_122"] = {"description": "double_122"}
        columnInfoMap["double_123"] = {"description": "double_123"}
        columnInfoMap["double_124"] = {"description": "double_124"}
        columnInfoMap["double_125"] = {"description": "double_125"}
        columnInfoMap["double_126"] = {"description": "double_126"}
        columnInfoMap["double_127"] = {"description": "double_127"}
        columnInfoMap["double_128"] = {"description": "double_128"}
        columnInfoMap["double_129"] = {"description": "double_129"}
        columnInfoMap["double_130"] = {"description": "double_130"}
        columnInfoMap["double_131"] = {"description": "double_131"}
        columnInfoMap["double_132"] = {"description": "double_132"}
        columnInfoMap["double_133"] = {"description": "double_133"}
        columnInfoMap["double_134"] = {"description": "double_134"}
        columnInfoMap["double_135"] = {"description": "double_135"}
        columnInfoMap["double_136"] = {"description": "double_136"}
        columnInfoMap["double_137"] = {"description": "double_137"}
        columnInfoMap["double_138"] = {"description": "double_138"}
        columnInfoMap["double_139"] = {"description": "double_139"}
        columnInfoMap["double_140"] = {"description": "double_140"}
        columnInfoMap["double_141"] = {"description": "double_141"}
        columnInfoMap["double_142"] = {"description": "double_142"}
        columnInfoMap["double_143"] = {"description": "double_143"}
        columnInfoMap["double_144"] = {"description": "double_144"}
        columnInfoMap["double_145"] = {"description": "double_145"}
        columnInfoMap["double_146"] = {"description": "double_146"}
        columnInfoMap["double_147"] = {"description": "double_147"}
        columnInfoMap["double_148"] = {"description": "double_148"}
        columnInfoMap["double_149"] = {"description": "double_149"}
        columnInfoMap["double_150"] = {"description": "double_150"}
        columnInfoMap["double_151"] = {"description": "double_151"}
        columnInfoMap["double_152"] = {"description": "double_152"}
        columnInfoMap["double_153"] = {"description": "double_153"}
        columnInfoMap["double_154"] = {"description": "double_154"}
        columnInfoMap["double_155"] = {"description": "double_155"}
        columnInfoMap["double_156"] = {"description": "double_156"}
        columnInfoMap["double_157"] = {"description": "double_157"}
        columnInfoMap["double_158"] = {"description": "double_158"}
        columnInfoMap["double_159"] = {"description": "double_159"}
        columnInfoMap["double_160"] = {"description": "double_160"}
        columnInfoMap["double_161"] = {"description": "double_161"}
        columnInfoMap["double_162"] = {"description": "double_162"}
        columnInfoMap["double_163"] = {"description": "double_163"}
        columnInfoMap["double_164"] = {"description": "double_164"}
        columnInfoMap["double_165"] = {"description": "double_165"}
        columnInfoMap["double_166"] = {"description": "double_166"}
        columnInfoMap["double_167"] = {"description": "double_167"}
        columnInfoMap["double_168"] = {"description": "double_168"}
        columnInfoMap["double_169"] = {"description": "double_169"}
        columnInfoMap["double_170"] = {"description": "double_170"}
        columnInfoMap["double_171"] = {"description": "double_171"}
        columnInfoMap["double_172"] = {"description": "double_172"}
        columnInfoMap["double_173"] = {"description": "double_173"}
        columnInfoMap["double_174"] = {"description": "double_174"}
        columnInfoMap["double_175"] = {"description": "double_175"}
        columnInfoMap["double_176"] = {"description": "double_176"}
        columnInfoMap["double_177"] = {"description": "double_177"}
        columnInfoMap["double_178"] = {"description": "double_178"}
        columnInfoMap["double_179"] = {"description": "double_179"}
        columnInfoMap["double_180"] = {"description": "double_180"}
        columnInfoMap["double_181"] = {"description": "double_181"}
        columnInfoMap["double_182"] = {"description": "double_182"}
        columnInfoMap["double_183"] = {"description": "double_183"}
        columnInfoMap["double_184"] = {"description": "double_184"}
        columnInfoMap["double_185"] = {"description": "double_185"}
        columnInfoMap["double_186"] = {"description": "double_186"}
        columnInfoMap["double_187"] = {"description": "double_187"}
        columnInfoMap["double_188"] = {"description": "double_188"}
        columnInfoMap["double_189"] = {"description": "double_189"}
        columnInfoMap["double_190"] = {"description": "double_190"}
        columnInfoMap["double_191"] = {"description": "double_191"}
        columnInfoMap["double_192"] = {"description": "double_192"}
        columnInfoMap["double_193"] = {"description": "double_193"}
        columnInfoMap["double_194"] = {"description": "double_194"}
        columnInfoMap["double_195"] = {"description": "double_195"}
        columnInfoMap["double_196"] = {"description": "double_196"}
        columnInfoMap["double_197"] = {"description": "double_197"}
        columnInfoMap["double_198"] = {"description": "double_198"}
        columnInfoMap["double_199"] = {"description": "double_199"}
        columnInfoMap["double_200"] = {"description": "double_200"}
        columnInfoMap["json_001"] = {"description": "json_001"}
        return columnInfoMap