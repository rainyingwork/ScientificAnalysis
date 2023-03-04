import os , copy
import Config
import OPSCommon as executeOPSCommon
import datetime
import threading
import time
from package.opsmanagement.common.tool.DceOPSTool import DceOPSTool

dceOPSTool = DceOPSTool()
dceOPSDF = dceOPSTool.getRunDCEOPSDF(runType = 'RunDCEOPS' , batchNumber = 'DCESystem')

maxOPSThread = 5

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
        thread.start() ; time.sleep(1)
        threadList.append(thread)
        threadAliveCount = maxOPSThread
        while threadAliveCount >= maxOPSThread:
            threadAliveCount = 0
            for thread in threadList:
                threadAliveCount += 1 if thread.is_alive() else 0
            if threadAliveCount >= maxOPSThread :
                print("Max OPSThread!! Wait 3 seconds...")
                time.sleep(3)
    for thread in threadList:
         thread.join()
