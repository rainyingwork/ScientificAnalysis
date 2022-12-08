import time
from hdfs import InsecureClient

class HDFSCtrl:

    def __init__(self, url=None, user=None, password=None, filePath=None):
        self.__url = url
        self.__user = user
        self.__password = password
        self.__filePath = filePath


    def listing(self, hdfsPath):
        hdfs_client = InsecureClient(self.__url, user=self.__user,root=self.__filePath)
        return hdfs_client.list(hdfsPath)


    def makeDir(self, hdfsPath):
        hdfs_client = InsecureClient(self.__url, user=self.__user,root=self.__filePath)
        hdfs_client.makedirs(hdfsPath)


    def deleteDir(self, hdfsPath):
        hdfs_client = InsecureClient(self.__url, user=self.__user, root=self.__filePath)
        hdfs_client.delete(hdfsPath,recursive=True, skip_trash=True)


    def uploadFile(self,localFile, hdfsFile):
        hdfs_client = InsecureClient(self.__url, user=self.__user,root=self.__filePath)
        try:
            hdfs_client.upload(hdfsFile, localFile, overwrite=True, cleanup=True)
        except:
            self.makeDir('/'.join(hdfsFile.split("/")[:-1]))
            time.sleep(0.5)
            hdfs_client.upload(hdfsFile, localFile, overwrite=True, cleanup=True)


    def downloadFile(self, hdfsFile, localFile):
        hdfs_client = InsecureClient(self.__url, user=self.__user,root=self.__filePath)
        hdfs_client.download(hdfsFile, localFile, overwrite=True)



