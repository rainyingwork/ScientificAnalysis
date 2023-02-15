from package.common.common.osbasic.SSHCtrl import SSHCtrl

class LoadBalanceSSHCtrl:

    def __init__(self,sshInfoList):
        self.sshCtrlArray = []
        for sshInfo in sshInfoList :
            sshCtrl = SSHCtrl(host=sshInfo["IP"], port=sshInfo["Port"], user=sshInfo["User"], passwd=sshInfo["Passwd"],printLog=False)
            self.sshCtrlArray.append(sshCtrl)

    def __del__(self):
        for sshCtrl in self.sshCtrlArray:
            del sshCtrl

    def executeAllSSH(self, commandArr):
        massageArr = []
        for commandStr in commandArr:
            for sshCtrl in self.sshCtrlArray :
                massageArr.append(sshCtrl.execCommand(commandStr))
        return massageArr

    def executeAllSSHByLowLoadSSHCtrl(self, commandArr):
        sshCtrl = self.getLowLoadSSHCtrl()
        massageArr = []
        for commandStr in commandArr:
            massageArr.append(sshCtrl.execCommand(commandStr))
        return massageArr

    def getLowLoadSSHCtrl(self):
        lowLoadSSHCtrl = None
        lowLoadCPU = 0.80
        for sshCtrl in self.sshCtrlArray:
            returnStr = sshCtrl.execCommand("cat /proc/loadavg")
            returnArr = returnStr.split(" ")
            loadCPU = float(returnArr[0]) if float(returnArr[0]) > float(returnArr[1]) else float(returnArr[1])
            if lowLoadCPU > loadCPU:
                lowLoadSSHCtrl = sshCtrl
                lowLoadCPU = loadCPU
        return lowLoadSSHCtrl


