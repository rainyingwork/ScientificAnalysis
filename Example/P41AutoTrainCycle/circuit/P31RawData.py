
class RawData() :

    @classmethod
    def R1_1_0(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R1_1_0"])
        functionVersionInfo["Version"] = "R1_1_0"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R1_1_1(self, functionInfo):
        import os, copy
        import pandas
        from dotenv import load_dotenv
        from package.common.common.database.PostgresCtrl import PostgresCtrl
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R1_1_1"])
        load_dotenv(dotenv_path="env/postgresql.env")
        postgresCtrl = PostgresCtrl(
            host=os.getenv("POSTGRES_HOST")
            , port=int(os.getenv("POSTGRES_POST"))
            , user=os.getenv("POSTGRES_USERNAME")
            , password=os.getenv("POSTGRES_PASSWORD")
            , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
            , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
        )
        sql = """
            DELETE FROM observationdata.analysisdata
            WHERE product = '[:Product]' AND project = '[:Project]' AND version = 'R1_1_1' AND dt = '[:DateNoLine]' ; 
            
            Insert into observationdata.analysisdata
            select 
                product , project ,  'R1_1_1' as version , dt
                , common_001 , null as common_002, null as common_003, null as common_004, null as common_005
                , null as common_006, null as common_007, null as common_008, null as common_009, null as common_010
                , null as common_011, null as common_012, null as common_013, null as common_014, null as common_015
                , 1 ::double precision as double_001
                , SUM(1)::double precision as double_002
                , ROUND(SUM(EXTRACT(EPOCH FROM (time_002 - time_001)))/(60*60*24),4)::double precision as double_003
                , null::double precision as double_004, null::double precision as double_005
                , null::double precision as double_006, null::double precision as double_007, null::double precision as double_008, null::double precision as double_009, null::double precision as double_010
                , null::double precision as double_011, null::double precision as double_012, null::double precision as double_013, null::double precision as double_014, null::double precision as double_015
                , null::double precision as double_016, null::double precision as double_017, null::double precision as double_018, null::double precision as double_019, null::double precision as double_020
                , null::double precision as double_021, null::double precision as double_022, null::double precision as double_023, null::double precision as double_024, null::double precision as double_025
                , null::double precision as double_026, null::double precision as double_027, null::double precision as double_028, null::double precision as double_029, null::double precision as double_030
                , null::double precision as double_031, null::double precision as double_032, null::double precision as double_033, null::double precision as double_034, null::double precision as double_035
                , null::double precision as double_036, null::double precision as double_037, null::double precision as double_038, null::double precision as double_039, null::double precision as double_040
                , null::double precision as double_041, null::double precision as double_042, null::double precision as double_043, null::double precision as double_044, null::double precision as double_045
                , null::double precision as double_046, null::double precision as double_047, null::double precision as double_048, null::double precision as double_049, null::double precision as double_050
                , null::double precision as double_051, null::double precision as double_052, null::double precision as double_053, null::double precision as double_054, null::double precision as double_055
                , null::double precision as double_056, null::double precision as double_057, null::double precision as double_058, null::double precision as double_059, null::double precision as double_060
                , null::double precision as double_061, null::double precision as double_062, null::double precision as double_063, null::double precision as double_064, null::double precision as double_065
                , null::double precision as double_066, null::double precision as double_067, null::double precision as double_068, null::double precision as double_069, null::double precision as double_070
                , null::double precision as double_071, null::double precision as double_072, null::double precision as double_073, null::double precision as double_074, null::double precision as double_075
                , null::double precision as double_076, null::double precision as double_077, null::double precision as double_078, null::double precision as double_079, null::double precision as double_080
                , null::double precision as double_081, null::double precision as double_082, null::double precision as double_083, null::double precision as double_084, null::double precision as double_085
                , null::double precision as double_086, null::double precision as double_087, null::double precision as double_088, null::double precision as double_089, null::double precision as double_090
                , null::double precision as double_091, null::double precision as double_092, null::double precision as double_093, null::double precision as double_094, null::double precision as double_095
                , null::double precision as double_096, null::double precision as double_097, null::double precision as double_098, null::double precision as double_099, null::double precision as double_100
                , null::double precision as double_101, null::double precision as double_102, null::double precision as double_103, null::double precision as double_104, null::double precision as double_105
                , null::double precision as double_106, null::double precision as double_107, null::double precision as double_108, null::double precision as double_109, null::double precision as double_110
                , null::double precision as double_111, null::double precision as double_112, null::double precision as double_113, null::double precision as double_114, null::double precision as double_115
                , null::double precision as double_116, null::double precision as double_117, null::double precision as double_118, null::double precision as double_119, null::double precision as double_120
                , null::double precision as double_121, null::double precision as double_122, null::double precision as double_123, null::double precision as double_124, null::double precision as double_125
                , null::double precision as double_126, null::double precision as double_127, null::double precision as double_128, null::double precision as double_129, null::double precision as double_130
                , null::double precision as double_131, null::double precision as double_132, null::double precision as double_133, null::double precision as double_134, null::double precision as double_135
                , null::double precision as double_136, null::double precision as double_137, null::double precision as double_138, null::double precision as double_139, null::double precision as double_140
                , null::double precision as double_141, null::double precision as double_142, null::double precision as double_143, null::double precision as double_144, null::double precision as double_145
                , null::double precision as double_146, null::double precision as double_147, null::double precision as double_148, null::double precision as double_149, null::double precision as double_150
                , null::double precision as double_151, null::double precision as double_152, null::double precision as double_153, null::double precision as double_154, null::double precision as double_155
                , null::double precision as double_156, null::double precision as double_157, null::double precision as double_158, null::double precision as double_159, null::double precision as double_160
                , null::double precision as double_161, null::double precision as double_162, null::double precision as double_163, null::double precision as double_164, null::double precision as double_165
                , null::double precision as double_166, null::double precision as double_167, null::double precision as double_168, null::double precision as double_169, null::double precision as double_170
                , null::double precision as double_171, null::double precision as double_172, null::double precision as double_173, null::double precision as double_174, null::double precision as double_175
                , null::double precision as double_176, null::double precision as double_177, null::double precision as double_178, null::double precision as double_179, null::double precision as double_180
                , null::double precision as double_181, null::double precision as double_182, null::double precision as double_183, null::double precision as double_184, null::double precision as double_185
                , null::double precision as double_186, null::double precision as double_187, null::double precision as double_188, null::double precision as double_189, null::double precision as double_190
                , null::double precision as double_191, null::double precision as double_192, null::double precision as double_193, null::double precision as double_194, null::double precision as double_195
                , null::double precision as double_196, null::double precision as double_197, null::double precision as double_198, null::double precision as double_199, null::double precision as double_200
            from observationdata.standarddata
            where 1 = 1
                and product = '[:Product]'
                and project = '[:Project]'  
                and tablename = '1102'
                and dt =  '[:DateNoLine]'
            group by 
                product 
                , project
                , dt
                , common_001 ; 
        """
        sql = sql.replace("[:Product]", functionInfo["Product"])
        sql = sql.replace("[:Project]", functionInfo["Project"])
        sql = sql.replace("[:DateLine]", functionVersionInfo["DataTime"])
        sql = sql.replace("[:DateNoLine]", functionVersionInfo["DataTime"].replace("-", ""))

        sqlStrArr = sql.split(";")[:-1]

        for sqlStr in sqlStrArr:
            postgresCtrl.executeSQL(sqlStr)

        return {"result": "OK"}, {}

    @classmethod
    def R0_1_X(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_1_X"])
        functionVersionInfo["Version"] = "R0_1_X"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

    @classmethod
    def R0_11_X(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_11_X"])
        functionVersionInfo["Version"] = "R0_11_X"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

    @classmethod
    def R0_12_X(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_12_X"])
        functionVersionInfo["Version"] = "R0_12_X"
        for dictKey in ["FunctionItemType", "MakeDataKeys", "MakeDataInfo"]:
            functionVersionInfo[dictKey] = functionInfo["ResultJson"][functionVersionInfo["DataVersion"]][dictKey]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_20_X(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_20_X"])
        functionVersionInfo["Version"] = "R0_20_X"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict

    @classmethod
    def R0_21_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_21_X"])
        functionVersionInfo["Version"] = "R0_21_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_22_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_22_X"])
        functionVersionInfo["Version"] = "R0_22_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_23_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_23_X"])
        functionVersionInfo["Version"] = "R0_23_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_24_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_24_X"])
        functionVersionInfo["Version"] = "R0_24_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_25_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_25_X"])
        functionVersionInfo["Version"] = "R0_25_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_26_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_26_X"])
        functionVersionInfo["Version"] = "R0_26_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_27_X(self, functionInfo):
        import copy
        from package.common.common.osbasic.GainObjectCtrl import GainObjectCtrl
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_27_X"])
        functionVersionInfo["Version"] = "R0_27_X"
        globalObject = GainObjectCtrl.getObjectsById(functionInfo["GlobalObject"])
        functionVersionInfo["ResultArr"] = globalObject[functionVersionInfo["DataVersion"]]["ResultArr"]
        resultObject, globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject, globalObjectDict

    @classmethod
    def R0_31_X(self, functionInfo):
        import copy
        from package.artificialintelligence.common.rawdata.RawDataFunction import RawDataFunction
        rawDataFunction = RawDataFunction()
        functionVersionInfo = copy.deepcopy(functionInfo["ParameterJson"]["R0_31_X"])
        functionVersionInfo["Version"] = "R0_31_X"
        resultObject , globalObjectDict = rawDataFunction.executionFunctionByFunctionType(functionVersionInfo)
        return resultObject , globalObjectDict