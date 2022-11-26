import os
import json
import datetime
from package.common.osbasic.CommonException import CommonException
from package.common.entity.EntityBase import EntityBase

class OPSVersionEntity (EntityBase) :

    def __init__(self):
        EntityBase.__init__(self)
        self.schemaName = "opsmanagement"
        self.tableName = "opsversion"
        self.tableInfoDF = self.postgresCtrl.getTableInfoDF("{}.{}".format(self.schemaName,self.tableName))

    def getOPSVersionByProductProjectOPSVersion(self, product, project, opsVersion):
        sql = """
            SELECT * 
            FROM opsmanagement.opsversion AA
            WHERE 1 = 1 
                AND AA.product = '[:Product]'
                AND AA.project = '[:Project]'
                AND AA.opsversion = '[:OPSVersion]'
                AND AA.deletetime is null 
           ORDER BY 
                AA.modifytime DESC 
           limit 1 
        """.replace('[:Product]', product).replace('[:Project]', project).replace('[:OPSVersion]', opsVersion)
        df = self.postgresCtrl.searchSQL(sql)
        if len(df) == 0 :
            raise CommonException("Not find {} {} {} OPSVersion".format(product,project,opsVersion))
        self.setEntity(df.iloc[0].to_dict())
        return self.entity

    def deleteOldOPSVersionByOPSInfo(self,opsInfo) :
        sql = """
        update opsmanagement.opsversion 
        set deletetime  = now()
        where 1 = 1 
            and product	 = '[:Product]'
            and project = '[:Project]'
            and deletetime is null 
            and opsversion = '[:OPSVersion]'
        """.replace('[:Product]', opsInfo["Product"]) \
            .replace('[:Project]', opsInfo["Project"]) \
            .replace('[:OPSVersion]', opsInfo["OPSVersion"])
        self.postgresCtrl.executeSQL(sql)

    def makeOPSVersionEntityByOPSInfo(self,opsInfo):
        entity = {}
        entity["createtime"] = datetime.datetime.now()
        entity["modifytime"] = datetime.datetime.now()
        entity["deletetime"] = None
        entity["opsversionid"] = opsInfo["OPSVersionId"]
        entity["product"] = opsInfo["Product"]
        entity["project"] = opsInfo["Project"]
        entity["opsversion"] = opsInfo["OPSVersion"]
        entity["opsorderjson"] = json.dumps(opsInfo["OPSOrderJson"],ensure_ascii=False) if "OPSOrderJson" in opsInfo.keys() else '{}'
        entity["parameterjson"] = json.dumps(opsInfo["ParameterJson"],ensure_ascii=False) if "ParameterJson" in opsInfo.keys() else '{}'
        entity["resultjson"] = json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}'
        return entity







