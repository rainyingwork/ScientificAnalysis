import os ,sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import Config
from package.common.common.osbasic.InputCtrl import InputCtrl
from package.opsmanagement.common.OPSCtrlLocal import OPSCtrl

opsCtrl = OPSCtrl()

def main(parametersData = {}):
    runType = ""
    product = ""
    project = ""
    opsVersion = ""
    opsRecordId = None
    opsOrderJson = {}
    parameterJson = {}
    resultJson = {}
    runFunctionArr = []

    if parametersData == {}:
        parametersData = InputCtrl.makeParametersData(sys.argv)

    for key in parametersData.keys():
        if key == "RunType":
            runType = parametersData[key][0]
        if key == "Product":
            product = parametersData[key][0]
        if key == "Project":
            project = parametersData[key][0]
        if key == "OPSVersion":
            opsVersion = parametersData[key][0]
        if key == "OPSOrderJson":
            opsOrderJson = parametersData[key]
        if key == "ParameterJson":
            parameterJson = parametersData[key]
        if key == "ResultJson":
            resultJson = parametersData[key]
        # ==================== DCE專用 ====================
        if key == "OPSRecordId":
            opsRecordId = parametersData[key][0]
        if key == "RunFunctionArr":
            runFunctionArr = parametersData[key]

    opsInfo = {}
    if runType == "RunOPS":
        opsInfo = makeOPSInfoByRunOPS(runType, product, project, opsVersion, opsRecordId, opsOrderJson, parameterJson,resultJson)
        opsCtrl.executeOPS(opsInfo)
        print("Finish Run OPS , Version is {} , OPSRecordID is {} ".format(opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))

def makeOPSInfoByRunOPS (runType, product, project, opsVersion,opsRecordId,opsOrderJson, parameterJson, resultJson) :
    # 只RunOPS但無須建立資料庫紀錄
    opsInfo = {
        "OPSRecordId": opsRecordId
        , "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersion": opsVersion
        , "OPSOrderJson": opsOrderJson
        , "ParameterJson": parameterJson
        , "ResultJson": resultJson
    }
    return opsInfo
