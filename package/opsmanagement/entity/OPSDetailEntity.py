import os
import json
import datetime
from package.common.osbasic.CommonException import CommonException
from package.common.entity.EntityBase import EntityBase


class OPSDetailEntity (EntityBase) :

    def __init__(self):
        EntityBase.__init__(self)
        self.schemaName = "opsmanagement"
        self.tableName = "opsdetail"
        self.tableInfoDF = self.postgresCtrl.getTableInfoDF("{}.{}".format(self.schemaName,self.tableName ))

    def makeOPSDetailEntityByFunctionInfo(self,functionInfo):
        entity = {}
        entity["createtime"] = datetime.datetime.now()
        entity["modifytime"] = datetime.datetime.now()
        entity["deletetime"] = None
        entity["opsrecord"] = functionInfo["OPSRecordId"]
        entity["exefunction"] = functionInfo["ExeFunction"]
        entity["parameterjson"] = json.dumps(functionInfo["ParameterJson"],ensure_ascii=False) if "ParameterJson" in functionInfo.keys() else '{}'
        entity["resultjson"] = json.dumps(functionInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in functionInfo.keys() else '{}'
        entity["state"] = "FINISH"
        return entity

    def isHaveOPSDetailEntityByOPSRecordAndExeFunctionAndState(self,opsRecordId , exeFunction , state = "FINISH"):
        sql = """
            SELECT * 
            FROM opsmanagement.opsdetail AA
            WHERE 1 = 1 
                AND AA.opsrecord = [:OPSRecordId]
                AND AA.exefunction = '[:ExeFunction]'
                AND AA.state = '[:State]'
                AND AA.deletetime is null 
           limit 1 
        """.replace('[:OPSRecordId]', str(opsRecordId)).replace('[:ExeFunction]', exeFunction).replace('[:State]', state)
        df = self.postgresCtrl.searchSQL(sql)
        if len(df) == 0:
            return False
        return True