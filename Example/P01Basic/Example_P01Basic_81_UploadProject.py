from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P01Basic")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P01Basic")
sshCtrl.uploadDirBySFTP("Example/P01Basic/circuit","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P01Basic/circuit")