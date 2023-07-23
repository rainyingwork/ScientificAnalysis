import json
import datetime
from package.common.common.entity.EntityBase import EntityBase

class RequestData (EntityBase) :

    def __init__(self):
        EntityBase.__init__(self)
        self.schemaName = "observationdata"
        self.tableName = "requestdata"
        self.tableInfoDF = self.postgresCtrl.getTableInfoDF("{}.{}".format(self.schemaName,self.tableName ))

