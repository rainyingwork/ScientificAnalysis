from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis")
sshCtrl.uploadDirBySFTP("package","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/package")
sshCtrl.uploadDirBySFTP("env","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/env")
sshCtrl.uploadFile("Config.py","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Config.py")
sshCtrl.uploadFile("OPSCommon.py","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/OPSCommon.py")