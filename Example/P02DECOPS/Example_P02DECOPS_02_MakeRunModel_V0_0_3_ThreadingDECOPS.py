import os , copy; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
import OPSCommon as executeOPSCommon
import datetime
import threading
import time


if __name__ == "__main__":

    opsRecordIdArr = [1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]
    opsRecordIdArr = [ 1011, 1012, 1013]
    threadList = []
    basicInfo = {
        "RunType": ["decops"]
        , "Product": ["Example"]
        , "Project": ["P02DECOPS"]
    }
    for opsRecordId in opsRecordIdArr:
        opsInfo = copy.deepcopy(basicInfo)
        opsInfo["OPSVersion"] = ["V0_0_3"]
        opsInfo["OPSRecordId"] = [opsRecordId]
        thread = threading.Thread(target=executeOPSCommon.main, args=(opsInfo,))
        thread.daemon = True
        thread.start(), time.sleep(10)
        threadList.append(thread)
    for thread in threadList:
        thread.join()





