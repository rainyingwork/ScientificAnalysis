
class AnalysisFunction():

    def __init__(self):
        pass

    @classmethod
    def getAnalysisColumnNameArr(self):
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
        floatColumnArr = [
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
        return floatColumnArr

    @classmethod
    def getJsonColumnArr(self):
        JsonColumnArr = ["json_001"]
        return JsonColumnArr

