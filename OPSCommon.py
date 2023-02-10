import os, sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import json
import Config
from package.common.common.osbasic.InputCtrl import InputCtrl
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
    if runType == "BuildOPS":
        opsInfo = makeOPSInfoByBuildOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        print("Finish Build OPS , Product is {} , Project is {} , Version is {} ".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"]))
    elif runType == "RunOPS":
        opsInfo = makeOPSInfoByRunOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        opsCtrl.executeOPS(opsInfo)
        opsRecordEntityCtrl.setColumnValue("state", "FINISH")
        opsRecordEntityCtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRecordEntityCtrl.updateEntity()
        print("Finish Run OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {} ".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "CreatDCEOPS":
        opsInfo = makeOPSInfoByCreatDCEOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        print("Finish Creat DCE OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "RunDCEOPS":
        opsInfo = makeOPSInfoByRunDCEOPS(runType, product, project, opsVersion, opsRecordId)
        opsCtrl.executeDCE(opsInfo)
        opsRecordEntityCtrl.setColumnValue("state", "FINISH")
        opsRecordEntityCtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRecordEntityCtrl.updateEntity()
        print("Finish Run DCE OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "RunOnlyFunc":
        opsInfo = makeOPSInfoByRunOnlyFunc(runType, product, project, opsVersion, opsRecordId)
        opsInfo["OPSOrderJson"]["RepOPSRecordId"] = opsRecordId
        opsInfo["OPSOrderJson"]["RunFunctionArr"] = runFunctionArr
        opsCtrl.executeOPS(opsInfo)
        print("Finish Run Only Func , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    return opsInfo

def makeOPSInfoByBuildOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
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

def makeOPSInfoByRunOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
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

def makeOPSInfoByCreatDCEOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    # 與makeOPSInfoByRunOPS模式一樣，只建立opsRecordEntity但不執行
    return makeOPSInfoByRunOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)

def makeOPSInfoByRunDCEOPS (runType, product, project, opsVersion,opsRecordId) :
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

def makeOPSInfoByRunOnlyFunc (runType, product, project, opsVersion,opsRecordId) :
    # 與makeOPSInfoByRunDCEOPS模式一樣，只跑單一方法
    return makeOPSInfoByRunDCEOPS (runType, product, project, opsVersion,opsRecordId)

if __name__ == "__main__":
    main()