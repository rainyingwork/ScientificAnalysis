from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P34PyTorch")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P34PyTorch")
sshCtrl.uploadDirBySFTP("Example/P34PyTorch/circuit","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P34PyTorch/circuit")
sshCtrl.uploadDirBySFTP("Example/P34PyTorch/package","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P34PyTorch/package")