from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps")
sshCtrl.uploadDirBySFTP("Example/P02DceOps/circuit","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps/circuit")