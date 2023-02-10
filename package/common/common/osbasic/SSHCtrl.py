import os
import paramiko
import stat
import traceback
from package.common.common.osbasic.BaseFunction import timethis
from dotenv import load_dotenv

class SSHCtrl:
    def __init__(self, host=None , port=None, user=None, passwd=None,pkey=None,env=None,timeout=30,printLog=True):
        if env == None :
            self.__host = host
            self.__port = port
            self.__user = user
            self.__passwd = passwd
            self.__pkey = pkey
        else :
            load_dotenv(dotenv_path=env)
            self.__host = os.getenv("SSH_IP")
            self.__port = int(os.getenv("SSH_PORT"))
            self.__user = os.getenv("SSH_USER")
            self.__passwd = os.getenv("SSH_PASSWD")
            self.__pkey = os.getenv("SSH_PKEY")
        self.__timeout = timeout
        self.__ssh = None
        self.__sftp = None
        self.__printLog = printLog
        self.connectSSH()
        self.connectSFTP()

    def __del__(self):
        if self.__ssh:
            self.__ssh.close()
        try:
            if self.__sftp:
                self.__sftp.close()
        except :
            pass

    def connectSFTP(self):
        try:
            transport = paramiko.Transport((self.__host, self.__port))
            if self.__pkey == None :
                transport.connect(username=self.__user, password=self.__passwd)
            else :
                key = paramiko.RSAKey.from_private_key_file(self.__pkey)
                transport.connect(username=self.__user, pkey=key)
            self.__sftp = paramiko.SFTPClient.from_transport(transport)
        except Exception as e:
            raise RuntimeError("sftp connect failed [%s]" % str(e))

    def connectSSH(self):
        try:
            self.__ssh = paramiko.SSHClient()
            self.__ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if self.__pkey == None :
                self.__ssh.connect(hostname=self.__host,
                                  port=self.__port,
                                  username=self.__user,
                                  password=self.__passwd,
                                  timeout=self.__timeout)
            else :
                key = paramiko.RSAKey.from_private_key_file(self.__pkey)
                self.__ssh.connect(hostname=self.__host,
                                  port=self.__port,
                                  username=self.__user,
                                  timeout=self.__timeout,
                                  pkey=key)
        except Exception as e:
            raise RuntimeError("ssh connect failed [%s]" % str(e))

    # common ----------------------------------------------------------------------------------------------------

    @staticmethod
    def isShellFile(fileName):
        return fileName.endswith(".sh")

    @staticmethod
    def isFileExist(fileName):
        try:
            with open(fileName, "r"):
                return True
        except Exception as e:
            return False

    # 檢測遠端的指令碼檔案和當前的指令碼檔案是否一致
    def checkFileConsistent(self, localFile, remoteFile):
        try:
            result = self.execCommand("find" + remoteFile)
            if len(result) == 0:
                self.uploadFile(localFile, remoteFile)
            else:
                localFileSize = os.path.getsize(localFile)
                result = self.execCommand("du -b" + remoteFile)
                remoteFileSize = int(result.split("\t")[0])
                if localFileSize != remoteFileSize:
                    self.uploadFile(localFile, remoteFile)
        except Exception as e:
            raise RuntimeError("check error [%s]" % str(e))

    # ssh ----------------------------------------------------------------------------------------------------

    # 通過ssh在遠端執行命令
    def execCommand(self, commandStr):
        try:
            stdin, stdout, stderr = self.__ssh.exec_command(commandStr, get_pty=True)
            if self.__printLog :
                for line in iter(stdout.readline, ""):
                    print(line, end="")
            return stdout.read().decode()
        except Exception as e:
            raise RuntimeError("Exec command [%s] failed" % str(commandStr))

    # 通過ssh在遠端特定的資料夾執行命令
    def execSSHCommand(self, cmd, path="~"):
        try:
            result = self.execCommand("cd " + path + ";" + cmd)
            if self.__printLog:
                print(result)
        except Exception:
            raise RuntimeError("exec cmd [%s] failed" % cmd)

    # 通過ssh在遠端特定的資料夾執行命令，並且回傳訊息
    def execSSHCommandReturn(self, cmd, path="~"):
        try:
            result = self.execCommand("cd " + path + ";" + cmd)
            return result
        except Exception:
            raise RuntimeError("exec cmd [%s] failed" % cmd)

    # 通過ssh使用sudo在遠端特定的資料夾執行命令
    def execSSHCommandBySudo(self, cmd, path="~"):
        self.execSSHCommand("sudo " + cmd, path)

    # 通過ssh使用sudo在遠端特定的資料夾執行命令，並且回傳訊息
    def execSSHCommandReturnBySudo(self, cmd, path="~"):
        self.execSSHCommandReturn(cmd, path)

    # 執行遠端的sh指令碼檔案，如果不一致，則上傳本地指令碼檔案
    def execSSHFile(self, localFile, remoteFile, execPath):
        try:
            if not self.isFileExist(localFile):
                raise RuntimeError("File [%s] not exist" % localFile)
            if not self.isShellFile(localFile):
                raise RuntimeError("File [%s] is not a shell file" % localFile)
            self.checkFileConsistent(localFile, remoteFile)
            result = self.execCommand("chmod +x " + remoteFile + "; cd" + execPath + ";/bin/bash " + remoteFile)
            if self.__printLog:
                print("exec shell result: ", result)
        except Exception as e:
            raise RuntimeError("ssh exec shell failed [%s]" % str(e))

    # sftp ----------------------------------------------------------------------------------------------------

    # 通過sftp上傳本地檔案到遠端
    def uploadFile(self, localFile, remoteFile):
        try:
            self.__sftp.put(localFile, remoteFile)
        except Exception as e:
            raise RuntimeError("upload failed [%s]" % str(e))

    # 通過sftp上傳本地檔案到遠端
    def downloadFile(self, remoteFile, localFile):
        try:
            self.__sftp.get(remoteFile, localFile)
        except Exception as e:
            raise RuntimeError("upload failed [%s]" % str(e))

    # 遞迴遠端所有目錄與文件
    def getAllFilesInRemoteDir(self, remoteDir):
        allFiles = list()
        files = self.__sftp.listdir_attr(remoteDir)
        for file in files:
            fileName = remoteDir + '/' + file.filename
            if stat.S_ISDIR(file.st_mode):  # 如果是文件遞迴處理
                allFiles.extend(self.getAllFilesInRemoteDir(self.__sftp, fileName))
            else:
                allFiles.append(fileName)
        return allFiles

    # 遞迴本地所有目錄與文件
    def getAllFilesInLocalDir(self, localDir):
        allFiles = list()
        for root, dirs, files in os.walk(localDir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                filename = filename.replace("\\","/")
                allFiles.append(filename)
        return allFiles

    # 本地文件夾上傳到遠端伺服器
    @timethis
    def uploadDirBySFTP(self, localDir,remoteDir):
        try:
            fileArr = self.getAllFilesInLocalDir(localDir)
            for file in fileArr:
                remoteFilename = remoteDir + file[len(localDir):]
                remotePath = remoteFilename.rsplit('/', maxsplit=1)[0]
                try:
                    self.__sftp.stat(remotePath)
                except:
                    self.execSSHCommand('mkdir -p %s' % remotePath)
                self.__sftp.put(file, remoteFilename)
                if self.__printLog:
                    print("Loacl " + str(file) + " file to remote " + str(remoteFilename) + " file.")
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())

    # 遠端伺服器下載到本地文件夾
    def downloadDirBySFTP(self, remoteDir, localDir):
        try:
            fileArr = self.getAllFilesInRemoteDir(remoteDir)
            for file in fileArr:
                localFileName = file.replace(remoteDir, localDir)
                localFilePath = os.path.dirname(localFileName)
                if not os.path.exists(localFilePath):
                    os.makedirs(localFilePath)
                self.__sftp.get(file, localFileName)
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())  #