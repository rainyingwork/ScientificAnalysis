import Config
from package.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
# sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis")
# sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis")
# sshCtrl.uploadDirBySFTP("package","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/package")
# sshCtrl.uploadDirBySFTP("env","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/env")
# sshCtrl.uploadDirBySFTP("common","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/common")
# sshCtrl.uploadFile("Config.py","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Config.py")
# sshCtrl.uploadFile("OPSCommon.py","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/OPSCommon.py")
sshCtrl.uploadFile("env/linux_ssh.env","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/env/ssh.env")
sshCtrl.uploadFile("env/linux_postgresql.env","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/env/postgresql.env")