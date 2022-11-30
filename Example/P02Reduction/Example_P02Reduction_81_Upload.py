import Config
from package.common.osbasic.SSHCtrl import SSHCtrl

sshCtrl = SSHCtrl(env="env/ssh.env")
sshCtrl.execSSHCommand("rm -rf /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P02Reduction")
sshCtrl.execSSHCommand("mkdir -p /mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P02Reduction")
sshCtrl.uploadDirBySFTP("Example/P02Reduction/circuit","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/Example/P02Reduction/circuit")

# sshCtrl.uploadFile("OPSCommon.py","/mfs/Docker/Python310/Volumes/Data/ScientificAnalysis/OPSCommon.py")
# sshCtrl.execSSHCommand("docker exec -it python310 python3 /Data/ScientificAnalysis/OPSCommon.py --RunType runfunc --Product Example --Project P02Reduction --OPSVersion V0_0_2 --OPSRecordId 924 --RunFunctionArr R0_0_1