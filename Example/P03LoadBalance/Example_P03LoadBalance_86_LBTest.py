import os
from dotenv import load_dotenv
from package.systemengineer.common.loadbalance.LoadBalanceSSHCtrl import LoadBalanceSSHCtrl
load_dotenv(dotenv_path="env/ssh.env")
sshInfoList = [
    {"IP":os.getenv("SSH_IP"),"Port":int(os.getenv("SSH_PORT")),"User":os.getenv("SSH_USER"),"Passwd":os.getenv("SSH_PASSWD")},
]
loadBalanceSSHCtrl = LoadBalanceSSHCtrl(sshInfoList)

print(loadBalanceSSHCtrl.executeAllSSH(["docker restart python39-cpu"]))
print(loadBalanceSSHCtrl.executeAllSSHByLowLoadSSHCtrl(["ls -al"]))
