import os
import ftplib

class FTPCtrl:
    bIsDir = False
    path = ""
    def __init__(self, host=None, port=None , user=None, passwd=None):
        self.__ftp = ftplib.FTP()
        self.__ftp.set_debuglevel(2)
        self.__ftp.connect(host, port)
        self.__ftp.login(user, passwd)
        print(self.__ftp.welcome)

    def __del__(self):
        self.__ftp.quit()

    # 上傳本地檔案到遠端
    def uploadFile(self, LocalFile, RemoteFile):
        if os.path.isfile(LocalFile) == False:
            return False
        file_handler = open(LocalFile, "rb")
        self.__ftp.storbinary('STOR %s' % RemoteFile, file_handler, 4096)
        file_handler.close()
        return True

    # 下載遠端檔案到本地
    def downloadFile(self,RemoteFile,LocalFile ):
        file_handler = open(LocalFile, 'wb')
        print(file_handler)
        self.__ftp.retrbinary("RETR %s" % (RemoteFile), file_handler.write)
        file_handler.close()
        return True

    # 上傳本地文件夾到遠端伺服器
    def uploadFileTree(self, LocalDir, RemoteDir):
        if os.path.isdir(LocalDir) == False:
            return False
        print("LocalDir:", LocalDir)
        LocalNames = os.listdir(LocalDir)
        print("list:", LocalNames)
        print(RemoteDir)
        self.__ftp.cwd(RemoteDir)
        for Local in LocalNames:
            src = os.path.join(LocalDir, Local)
            if os.path.isdir(src):
                self.uploadFileTree(src, Local)
            else:
                self.uploadFile(src, Local)

        self.__ftp.cwd("..")
        return

    # 下載遠端文件夾到本地伺服器
    def downloadFileTree(self, RemoteDir , LocalDir):
        print("remoteDir:", RemoteDir)
        if os.path.isdir(LocalDir) == False:
            os.makedirs(LocalDir)
        self.__ftp.cwd(RemoteDir)
        RemoteNames = self.__ftp.nlst()
        print("RemoteNames", RemoteNames)
        print(self.__ftp.nlst("/del1"))
        for file in RemoteNames:
            Local = os.path.join(LocalDir, file)
            if self.isDir(file):
                self.downloadFileTree(Local, file)
            else:
                self.downloadFile(Local, file)
        self.__ftp.cwd("..")
        return

    # 尚未修正====================================================================================================

    def show(self, list):
        result = list.lower().split(" ")
        if self.path in result and "<dir>" in result:
            self.bIsDir = True

    def isDir(self, path):
        self.bIsDir = False
        self.path = path
        # this ues callback function ,that will change bIsDir value
        self.__ftp.retrlines('LIST', self.show)
        return self.bIsDir

    def list(self, path):
        return self.__ftp.nlst(path)

