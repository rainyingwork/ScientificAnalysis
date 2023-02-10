# PostgresCtrl
##### Postgresql 操作相關 function  
```python
def init(host,port,user,password,database,schema) 
```
* PostgresCtrl 初始化
  * host 資料庫IP  
  * port 資料庫Port
  * user 使用者名稱
  * password 使用者密碼  
  * database 資料庫名稱
  * schema 資料庫綱要
```python
def executeSQL(sql)
```
* 執行SQL
  * sql 資料庫語法
```python
def searchSQL(sql)
```
* 查詢SQL，回傳Dataframe格式的資料
  * sql 資料庫語法
```python
def insertDataList(tableFullName,insertTableInfoDF,insertDataDF,insertMaxCount)
```
* 將Dataframe，使用語法方式，直接塞入資料庫指定Table
  * tableFullName 資料表名稱
  * insertTableInfoDF 資料表欄位訊息
  * insertDataDF 塞入資料
  * insertMaxCount 一次資料庫塞入最大比數
```python
def insertDataListByIO(tableFullName,insertTableInfoDF,insertDataDF,ifExists)
```
* 將Dataframe，使用IO方式，直接塞入資料庫指定Table
  * tableFullName 資料表名稱
  * insertTableInfoDF 資料表欄位訊息
  * insertDataDF 新增資料
  * insertMaxCount 一次資料庫塞入最大比數
```python
def updateDataList(tableFullName,updateTableInfoDF,updateData)
```
* 將Dataframe，使用語法方式，更新資料庫(必須指定TableID)
  * tableFullName 資料表名稱
  * updateTableInfoDF 資料表欄位訊息
  * updateData 更新資料
```python
def isTableExist(tableName)
```
* 確認資料表是否存在，
  * tableName 資料表名稱
```python
def getTableInfoDF(tableName)
```
* 取得資料表資訊
  * tableName 資料表名稱





    
