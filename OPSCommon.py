import os, sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import json
from package.common.osbasic.InputCtrl import InputCtrl
from package.opsmanagement.common.OPSCtrl import OPSCtrl
from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity
from package.opsmanagement.common.entity.OPSRecordEntity import OPSRecordEntity

opsCtrl = OPSCtrl()
opsVersionEntityCtrl = OPSVersionEntity()
opsRecordEntityCtrl = OPSRecordEntity()
inputCtrl = InputCtrl()

def main(parametersData = {},):

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
        parametersData = inputCtrl.makeParametersData(sys.argv)

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
    if runType == "buildops" :
        opsInfo = makeBuildOPSInfo(runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)
    elif runType in ["runops","creatdecops"] :
        opsInfo = makeRunOPSInfo(runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)
    elif runType in ["decops","runfunc"]:
        opsInfo = makeRunFuncOPSInfo (runType, product, project, opsVersion,opsRecordId)

    if runType == "buildops":
        print("Finish Build OPS , Version is {} ".format(opsInfo["OPSVersion"]))
    elif runType == "runops":
        opsCtrl.executeOPS(opsInfo)
        opsRecordEntityCtrl.setColumnValue("state", "FINISH")
        opsRecordEntityCtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRecordEntityCtrl.updateEntity()
        print("Finish Run OPS , Version is {} , OPSRecordID is {} ".format(opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "creatdecops":
        print("Finish Creat DCE OPS , Version is {} , OPSRecordID is {}".format(opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "decops":
        opsCtrl.executeDCE(opsInfo)
        opsRecordEntityCtrl.setColumnValue("state", "FINISH")
        opsRecordEntityCtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRecordEntityCtrl.updateEntity()
        print("Finish DCE OPS , Version is {} , OPSRecordID is {}".format(opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "runfunc":
        opsInfo["OPSOrderJson"]["RepOPSRecordId"] = opsRecordId
        opsInfo["OPSOrderJson"]["RunFunctionArr"] = runFunctionArr
        opsCtrl.executeOPS(opsInfo)
        print("Finish DCE OPS , Version is {} , OPSRecordID is {}".format(opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    return opsInfo

def makeBuildOPSInfo (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    opsInfo = {
        "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersionId": opsVersionEntityCtrl  .getNextPrimaryKeyId()
        , "OPSVersion": opsVersion
        , "OPSOrderJson": opsOrderJson
        , "ParameterJson": parameterJson
        , "ResultJson": resultJson
    }
    opsVersionEntityCtrl.setEntity(opsVersionEntityCtrl.makeOPSVersionEntityByOPSInfo(opsInfo))
    opsVersionEntityCtrl.deleteOldOPSVersionByOPSInfo(opsInfo)
    opsVersionEntityCtrl.insertEntity()
    return opsInfo

def makeRunOPSInfo (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    versionEntity = opsVersionEntityCtrl.getOPSVersionByProductProjectOPSVersion(product, project, opsVersion)
    opsInfo = {
        "OPSVersionId": versionEntity["opsversionid"]
        , "OPSRecordId": opsRecordEntityCtrl.getNextPrimaryKeyId()
        , "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersion": opsVersion
        , "OPSOrderJson": json.loads(versionEntity['opsorderjson']) if opsOrderJson == {} else opsOrderJson
        , "ParameterJson": json.loads(versionEntity['parameterjson']) if parameterJson == {} else parameterJson
        , "ResultJson": json.loads(versionEntity['resultjson']) if resultJson == {} else resultJson
    }
    opsRecordEntityCtrl.setEntity(opsRecordEntityCtrl.makeOPSRecordEntityByOPSInfo(opsInfo))
    opsRecordEntityCtrl.setColumnValue("state", "RUN")
    opsRecordEntityCtrl.insertEntity()
    return opsInfo

def makeRunFuncOPSInfo (runType, product, project, opsVersion,opsRecordId) :
    recordEntity = opsRecordEntityCtrl.getEntityByPrimaryKeyId(opsRecordId)
    opsInfo = {
        "OPSVersionId": recordEntity["opsversion"]
        , "OPSRecordId": recordEntity["opsrecordid"]
        , "RunType": runType
        , "Product": product
        , "Project": project
        , "OPSVersion": opsVersion
        , "OPSOrderJson": json.loads(recordEntity['opsorderjson'])
        , "ParameterJson": json.loads(recordEntity['parameterjson'])
        , "ResultJson": json.loads(recordEntity['resultjson'])
    }
    return opsInfo

if __name__ == "__main__":
    main()