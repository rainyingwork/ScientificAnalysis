
DROP TABLE IF EXISTS observationdata.analysisdata ;

CREATE TABLE IF NOT EXISTS observationdata.analysisdata (
    product text
    , project text
    , version text
    , dt text
    , common_001 text
    , common_002 text
    , common_003 text
    , common_004 text
    , common_005 text
    , common_006 text
    , common_007 text
    , common_008 text
    , common_009 text
    , common_010 text
    , common_011 text
    , common_012 text
    , common_013 text
    , common_014 text
    , common_015 text
    , double_001 double precision
    , double_002 double precision
    , double_003 double precision
    , double_004 double precision
    , double_005 double precision
    , double_006 double precision
    , double_007 double precision
    , double_008 double precision
    , double_009 double precision
    , double_010 double precision
    , double_011 double precision
    , double_012 double precision
    , double_013 double precision
    , double_014 double precision
    , double_015 double precision
    , double_016 double precision
    , double_017 double precision
    , double_018 double precision
    , double_019 double precision
    , double_020 double precision
    , double_021 double precision
    , double_022 double precision
    , double_023 double precision
    , double_024 double precision
    , double_025 double precision
    , double_026 double precision
    , double_027 double precision
    , double_028 double precision
    , double_029 double precision
    , double_030 double precision
    , double_031 double precision
    , double_032 double precision
    , double_033 double precision
    , double_034 double precision
    , double_035 double precision
    , double_036 double precision
    , double_037 double precision
    , double_038 double precision
    , double_039 double precision
    , double_040 double precision
    , double_041 double precision
    , double_042 double precision
    , double_043 double precision
    , double_044 double precision
    , double_045 double precision
    , double_046 double precision
    , double_047 double precision
    , double_048 double precision
    , double_049 double precision
    , double_050 double precision
    , double_051 double precision
    , double_052 double precision
    , double_053 double precision
    , double_054 double precision
    , double_055 double precision
    , double_056 double precision
    , double_057 double precision
    , double_058 double precision
    , double_059 double precision
    , double_060 double precision
    , double_061 double precision
    , double_062 double precision
    , double_063 double precision
    , double_064 double precision
    , double_065 double precision
    , double_066 double precision
    , double_067 double precision
    , double_068 double precision
    , double_069 double precision
    , double_070 double precision
    , double_071 double precision
    , double_072 double precision
    , double_073 double precision
    , double_074 double precision
    , double_075 double precision
    , double_076 double precision
    , double_077 double precision
    , double_078 double precision
    , double_079 double precision
    , double_080 double precision
    , double_081 double precision
    , double_082 double precision
    , double_083 double precision
    , double_084 double precision
    , double_085 double precision
    , double_086 double precision
    , double_087 double precision
    , double_088 double precision
    , double_089 double precision
    , double_090 double precision
    , double_091 double precision
    , double_092 double precision
    , double_093 double precision
    , double_094 double precision
    , double_095 double precision
    , double_096 double precision
    , double_097 double precision
    , double_098 double precision
    , double_099 double precision
    , double_100 double precision
    , double_101 double precision
    , double_102 double precision
    , double_103 double precision
    , double_104 double precision
    , double_105 double precision
    , double_106 double precision
    , double_107 double precision
    , double_108 double precision
    , double_109 double precision
    , double_110 double precision
    , double_111 double precision
    , double_112 double precision
    , double_113 double precision
    , double_114 double precision
    , double_115 double precision
    , double_116 double precision
    , double_117 double precision
    , double_118 double precision
    , double_119 double precision
    , double_120 double precision
    , double_121 double precision
    , double_122 double precision
    , double_123 double precision
    , double_124 double precision
    , double_125 double precision
    , double_126 double precision
    , double_127 double precision
    , double_128 double precision
    , double_129 double precision
    , double_130 double precision
    , double_131 double precision
    , double_132 double precision
    , double_133 double precision
    , double_134 double precision
    , double_135 double precision
    , double_136 double precision
    , double_137 double precision
    , double_138 double precision
    , double_139 double precision
    , double_140 double precision
    , double_141 double precision
    , double_142 double precision
    , double_143 double precision
    , double_144 double precision
    , double_145 double precision
    , double_146 double precision
    , double_147 double precision
    , double_148 double precision
    , double_149 double precision
    , double_150 double precision
    , double_151 double precision
    , double_152 double precision
    , double_153 double precision
    , double_154 double precision
    , double_155 double precision
    , double_156 double precision
    , double_157 double precision
    , double_158 double precision
    , double_159 double precision
    , double_160 double precision
    , double_161 double precision
    , double_162 double precision
    , double_163 double precision
    , double_164 double precision
    , double_165 double precision
    , double_166 double precision
    , double_167 double precision
    , double_168 double precision
    , double_169 double precision
    , double_170 double precision
    , double_171 double precision
    , double_172 double precision
    , double_173 double precision
    , double_174 double precision
    , double_175 double precision
    , double_176 double precision
    , double_177 double precision
    , double_178 double precision
    , double_179 double precision
    , double_180 double precision
    , double_181 double precision
    , double_182 double precision
    , double_183 double precision
    , double_184 double precision
    , double_185 double precision
    , double_186 double precision
    , double_187 double precision
    , double_188 double precision
    , double_189 double precision
    , double_190 double precision
    , double_191 double precision
    , double_192 double precision
    , double_193 double precision
    , double_194 double precision
    , double_195 double precision
    , double_196 double precision
    , double_197 double precision
    , double_198 double precision
    , double_199 double precision
    , double_200 double precision
    , json_001 text
) --PARTITION BY RANGE (product , project ,version,  dt);

CREATE INDEX analysisdata_index ON observationdata.analysisdata (product,project,version,dt) ;