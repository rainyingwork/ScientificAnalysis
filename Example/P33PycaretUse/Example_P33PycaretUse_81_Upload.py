import Config
from package.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P33PycaretUse")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P33PycaretUse")
sshCtrl.uploadDirBySFTP("Example/P33PycaretUse/circuit","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P33PycaretUse/circuit")
