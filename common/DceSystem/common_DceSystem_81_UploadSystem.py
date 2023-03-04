from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/common/DceSystem")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/common/DceSystem")
sshCtrl.uploadFile("common/DceSystem/common_DceSystem_11_CreatDceOPS.py","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/common_DceSystem_11_CreatDceOPS.py")
sshCtrl.uploadFile("common/DceSystem/common_DceSystem_12_RunDceOPS.py","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/common_DceSystem_12_RunDceOPS.py")
sshCtrl.uploadFile("common/DceSystem/Example_P34PyTorch_11_CreatDceOPS.py","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example_P34PyTorch_11_CreatDceOPS.py")
