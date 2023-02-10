import os , copy
import Config
import OPSCommon as executeOPSCommon
import datetime
import threading
import time


if __name__ == "__main__":

    opsRecordIdArr = [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    threadList = []
    basicInfo = {
        "RunType": ["RunDCEOPS"],
        "Product": ["Example"],
        "Project": ["P02DceOps"],
    }
    for opsRecordId in opsRecordIdArr:
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["OPSRecordId"] = [opsRecordId]
        thread = threading.Thread(target=executeOPSCommon.main, args=(opsInfo,))
        thread.daemon = True
        thread.start()
        threadList.append(thread)
        threadAliveCount = 10
        while threadAliveCount >= 10:
            time.sleep(1)
            threadAliveCount = 0
            for thread in threadList:
                threadAliveCount += 1 if thread.is_alive() else 0
    for thread in threadList:
        thread.join()





