import os , copy
import Config
import OPSCommon as executeOPSCommon
import datetime
import threading
import time
from package.opsmanagement.common.tool.DceOPSTool import DceOPSTool

dceOPSTool = DceOPSTool()
dceOPSDF = dceOPSTool.getRunDCEOPSDF(runType = 'RunDCEOPS' , batchNumber = '202211292300')

if __name__ == "__main__":
    threadList = []
    for index , row in dceOPSDF.iterrows() :
        opsInfo = {
            "RunType": ["RunDCEOPS"],
            "Product": [row['product']],
            "Project": [row['project']],
            "OPSVersion": [row['opsversion']],
            "OPSRecordId": [int(row['opsrecordid'])],
        }
        thread = threading.Thread(target=executeOPSCommon.main, args=(opsInfo,))
        thread.daemon = True
        thread.start()
        threadList.append(thread)
        threadAliveCount = 5
        while threadAliveCount >= 5:
            time.sleep(1)
            threadAliveCount = 0
            for thread in threadList:
                threadAliveCount += 1 if thread.is_alive() else 0
    for thread in threadList:
         thread.join()
