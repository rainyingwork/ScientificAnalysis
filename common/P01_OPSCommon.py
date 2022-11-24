import os ,sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import re , copy , json , pprint
import time , datetime
from package.common.osbasic.InputCtrl import InputCtrl
from package.opsmanagement.OPSCtrl import OPSCtrl
from package.opsmanagement.entity.OPSVersionEntity import OPSVersionEntity
from package.opsmanagement.entity.OPSRecordEntity import OPSRecordEntity

opsCtrl = OPSCtrl()
opsVersionEntityCtrl = OPSVersionEntity()
opsRecordEntityCtrl = OPSRecordEntity()

def main(parametersData = {}):
    runType = ""
    product = ""
    project = ""
    opsVersion = ""
    opsOrderJson = {}
    parameterJson = {}
    resultJson = {}

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

    opsInfo = {}
    if runType == "buildops" :
        opsInfo = makeBuildOPSInfo(runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)
    elif runType == "runops" :
        opsInfo = makeRunOPSInfo(runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)

    if runType == "buildops":
        opsVersionEntityCtrl.setEntity(opsVersionEntityCtrl.makeOPSVersionEntityByOPSInfo(opsInfo))
        opsVersionEntityCtrl.deleteOldOPSVersionByOPSInfo(opsInfo)
        opsVersionEntityCtrl.insertEntity()
        print("finish build ops , version is {} ".format(opsInfo["OPSVersion"]))
    elif runType == "runops":
        opsRecordEntityCtrl.setEntity(opsRecordEntityCtrl.makeOPSRecordEntityByOPSInfo(opsInfo))
        opsRecordEntityCtrl.setColumnValue("state", "RUN")
        opsRecordEntityCtrl.insertEntity()
        opsCtrl.executeOPS(opsInfo)
        time.sleep(1)
        opsRecordEntityCtrl.setColumnValue("state", "FINISH")
        opsRecordEntityCtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRecordEntityCtrl.updateEntity()
        print("finish run ops , version is {} ".format(opsInfo["OPSVersion"]))

def makeBuildOPSInfo (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    opsInfo = {
        "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersion": opsVersion
        , "OPSOrderJson": opsOrderJson
        , "ParameterJson": parameterJson
        , "ResultJson": resultJson
    }
    return opsInfo

def makeRunOPSInfo (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    modelVersionEntity = opsVersionEntityCtrl.getOPSVersionByProductProjectOPSVersion(product, project, opsVersion)
    opsInfo = {
        "OPSVersionId": modelVersionEntity["opsversionid"]
        , "OPSRecordId": opsRecordEntityCtrl.getNextPrimaryKeyId()
        , "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersion": opsVersion
        , "OPSOrderJson": json.loads(modelVersionEntity['opsorderjson']) if opsOrderJson == {} else opsOrderJson
        , "ParameterJson": json.loads(modelVersionEntity['parameterjson']) if parameterJson == {} else parameterJson
        , "ResultJson": json.loads(modelVersionEntity['resultjson']) if resultJson == {} else resultJson
    }

    return opsInfo
