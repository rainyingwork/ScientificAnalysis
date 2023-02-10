from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P36Pycaret")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P36Pycaret")
sshCtrl.uploadDirBySFTP("Example/P36Pycaret/circuit","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P36Pycaret/circuit")
