import json
import datetime
from package.common.common.entity.EntityBase import EntityBase


class OPSRecordEntity (EntityBase) :

    def __init__(self):
        EntityBase.__init__(self)
        self.schemaName = "opsmanagement"
        self.tableName = "opsrecord"
        self.tableInfoDF = self.postgresCtrl.getTableInfoDF("{}.{}".format(self.schemaName,self.tableName ))

    def makeOPSRecordEntityByOPSInfo(self,opsInfo):
        entity = {}
        entity["createtime"] = datetime.datetime.now()
        entity["modifytime"] = datetime.datetime.now()
        entity["deletetime"] = None
        entity["opsrecordid"] = opsInfo["OPSRecordId"]
        entity["opsversion"] = opsInfo["OPSVersionId"]
        entity["product"] = opsInfo["Product"]
        entity["project"] = opsInfo["Project"]
        entity["opsorderjson"] = json.dumps(opsInfo["OPSOrderJson"],ensure_ascii=False) if "OPSOrderJson" in opsInfo.keys() else '{}'
        entity["parameterjson"] = json.dumps(opsInfo["ParameterJson"],ensure_ascii=False) if "ParameterJson" in opsInfo.keys() else '{}'
        entity["resultjson"] = json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}'
        return entity

