from package.common.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps")
sshCtrl.uploadDirBySFTP("Example/P02DceOps/circuit","/mfs/Docker/Python39/Volumes/Data/ScientificAnalysis/Example/P02DceOps/circuit")

# sshCtrl.uploadFile("OPSCommon.py","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/OPSCommon.py")
# sshCtrl.execSSHCommand("docker exec -it python310 python3 /Data/ScientificAnalysis/OPSCommon.py --RunType runfunc --Product Example --Project P02DceOps --OPSVersion V0_0_2 --OPSRecordId 924 --RunFunctionArr R0_0_1