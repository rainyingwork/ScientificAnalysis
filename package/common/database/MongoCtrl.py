import pymongo

class MongoCtrl(object):
    # 初始化
    def __init__(self, db_ip, db_port, db_name, table_name):
        self.db_ip = db_ip
        self.db_port = db_port
        self.db_name = db_name
        self.table_name = table_name
        self.conn = pymongo.MongoClient(host=self.db_ip, port=self.db_port)
        self.db = self.conn[self.db_name]
        self.table = self.db[self.table_name]

    # 新增資料
    def insertData(self, kv_dict):
        return self.table.insert(kv_dict)

    # 更新資料
    def updateData(self, kv_dict , query ):
        ret = self.table.update_many(
            query,
            {
                "$set": kv_dict,
            }
        )
        if not ret.matched_count or ret.matched_count == 0:
            self.insert(kv_dict)
        elif ret.matched_count and ret.matched_count > 1:
            self.delete(query)
            self.insert(kv_dict)

    # 刪除資料
    def deleteData(self, query):
        return self.table.delete_many(query)

    # 尋找資料_單筆
    def findOne(self, query):
        return self.table.find_one(query, projection={"_id": False})

    # 尋找資料_多筆
    def findMany(self, query):
        return self.table.find(query)





