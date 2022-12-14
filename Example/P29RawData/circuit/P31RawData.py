import os
import copy , pprint
from dotenv import load_dotenv
from package.common.database.PostgresCtrl import PostgresCtrl

class RawData() :

    @classmethod
    def R0_0_0(self, functionInfo):
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_0"])
        functionVersionInfo["Version"] = "R0_0_0"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict


    @classmethod
    def R0_0_1(self, functionInfo):
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_0_1"])
        functionVersionInfo["Version"] = "R0_0_1"
        functionVersionInfo["SQLStrs"] = """
        DELETE FROM observationdata.analysisdata AA
        WHERE 1 = 1 
            AND AA.product = '[:Product]'
            AND AA.project = '[:Project]'
            AND AA.version = '[:FunctionVersion]'
            AND AA.dt = '[:DateNoLine]' ;
            
        INSERT INTO observationdata.analysisdata
        SELECT 
            '[:Product]' as product	
            , '[:Project]' as project	
            , '[:FunctionVersion]' as version	
            , '[:DateNoLine]' as dt
            , common_001 as common_001
            , null as common_002 
            , null as common_003 
            , null as common_004 
            , null as common_005 
            , null as common_006 
            , null as common_007 
            , null as common_008 
            , null as common_009 
            , null as common_010 
            , null as common_011 
            , null as common_012 
            , null as common_013 
            , null as common_014 
            , null as common_015 
            , case when string_001 = 'CH' then 1 else 0 end as double_001  
            , case when string_001 = 'MM' then 1 else 0 end as double_002  
            , case when string_002 = 'Yes' then 1 else 0 end as double_003 	
            , integer_001 as double_004
            , integer_002 as double_005
            , integer_003 as double_006	
            , integer_004 as double_007	
            , integer_005 as double_008	  
            , double_001 as double_009 	
            , double_002 as double_010	 	
            , double_003 as double_011	 	
            , double_004 as double_012	
            , double_005 as double_013	
            , double_006 as double_014	
            , double_007 as double_015	
            , double_008 as double_016	
            , double_009 as double_017	
            , double_010 as double_018
            , null as double_019 
            , null as double_020 
            , null as double_021 
            , null as double_022 
            , null as double_023 
            , null as double_024 
            , null as double_025 
            , null as double_026 
            , null as double_027 
            , null as double_028 
            , null as double_029 
            , null as double_030 
            , null as double_031 
            , null as double_032 
            , null as double_033 
            , null as double_034 
            , null as double_035 
            , null as double_036 
            , null as double_037 
            , null as double_038 
            , null as double_039 
            , null as double_040 
            , null as double_041 
            , null as double_042 
            , null as double_043 
            , null as double_044 
            , null as double_045 
            , null as double_046 
            , null as double_047 
            , null as double_048 
            , null as double_049 
            , null as double_050 
            , null as double_051 
            , null as double_052 
            , null as double_053 
            , null as double_054 
            , null as double_055 
            , null as double_056 
            , null as double_057 
            , null as double_058 
            , null as double_059 
            , null as double_060 
            , null as double_061 
            , null as double_062 
            , null as double_063 
            , null as double_064 
            , null as double_065 
            , null as double_066 
            , null as double_067 
            , null as double_068 
            , null as double_069 
            , null as double_070 
            , null as double_071 
            , null as double_072 
            , null as double_073 
            , null as double_074 
            , null as double_075 
            , null as double_076 
            , null as double_077 
            , null as double_078 
            , null as double_079 
            , null as double_080 
            , null as double_081 
            , null as double_082 
            , null as double_083 
            , null as double_084 
            , null as double_085 
            , null as double_086 
            , null as double_087 
            , null as double_088 
            , null as double_089 
            , null as double_090 
            , null as double_091 
            , null as double_092 
            , null as double_093 
            , null as double_094 
            , null as double_095 
            , null as double_096 
            , null as double_097 
            , null as double_098 
            , null as double_099 
            , null as double_100 
            , null as double_101 
            , null as double_102 
            , null as double_103 
            , null as double_104 
            , null as double_105 
            , null as double_106 
            , null as double_107 
            , null as double_108 
            , null as double_109 
            , null as double_110 
            , null as double_111 
            , null as double_112 
            , null as double_113 
            , null as double_114 
            , null as double_115 
            , null as double_116 
            , null as double_117 
            , null as double_118 
            , null as double_119 
            , null as double_120 
            , null as double_121 
            , null as double_122 
            , null as double_123 
            , null as double_124 
            , null as double_125 
            , null as double_126 
            , null as double_127 
            , null as double_128 
            , null as double_129 
            , null as double_130 
            , null as double_131 
            , null as double_132 
            , null as double_133 
            , null as double_134 
            , null as double_135 
            , null as double_136 
            , null as double_137 
            , null as double_138 
            , null as double_139 
            , null as double_140 
            , null as double_141 
            , null as double_142 
            , null as double_143 
            , null as double_144 
            , null as double_145 
            , null as double_146 
            , null as double_147 
            , null as double_148 
            , null as double_149 
            , null as double_150 
            , null as double_151 
            , null as double_152 
            , null as double_153 
            , null as double_154 
            , null as double_155 
            , null as double_156 
            , null as double_157 
            , null as double_158 
            , null as double_159 
            , null as double_160 
            , null as double_161 
            , null as double_162 
            , null as double_163 
            , null as double_164 
            , null as double_165 
            , null as double_166 
            , null as double_167 
            , null as double_168 
            , null as double_169 
            , null as double_170 
            , null as double_171 
            , null as double_172 
            , null as double_173 
            , null as double_174 
            , null as double_175 
            , null as double_176 
            , null as double_177 
            , null as double_178 
            , null as double_179 
            , null as double_180 
            , null as double_181 
            , null as double_182 
            , null as double_183 
            , null as double_184 
            , null as double_185 
            , null as double_186 
            , null as double_187 
            , null as double_188 
            , null as double_189 
            , null as double_190 
            , null as double_191 
            , null as double_192 
            , null as double_193 
            , null as double_194 
            , null as double_195 
            , null as double_196 
            , null as double_197 
            , null as double_198 
            , null as double_199 
            , null as double_200 
            , null as json_001 
        FROM observationdata.standarddata AA 
        WHERE 1 = 1 
            AND AA.product = 'Example'
            AND AA.project = 'P23Standard'
            AND AA.tablename = 'S0_0_1'
            AND AA.dt = '[:DateNoLine]' ; 
        """
        functionVersionInfo["SQLReplaceArr"] = rawDataFunction.getCommonSQLReplaceArr(functionInfo,functionVersionInfo)
        resultObject  , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

