import os, sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import json
import Config
from package.common.common.osbasic.InputCtrl import InputCtrl
from package.opsmanagement.common.OPSCtrl import OPSCtrl
from package.opsmanagement.common.entity.OPSVersionEntity import OPSVersionEntity
from package.opsmanagement.common.entity.OPSRecordEntity import OPSRecordEntity

opsCtrl = OPSCtrl()
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
        opsVECtrl , opsRECtrl , opsInfo = makeOPSInfoByBuildOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        print("Finish Build OPS , Product is {} , Project is {} , Version is {} ".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"]))
    elif runType == "RunOPS":
        opsVECtrl , opsRECtrl , opsInfo = makeOPSInfoByRunOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        opsCtrl.executeOPS(opsInfo)
        opsRECtrl.setColumnValue("state", "FINISH")
        opsRECtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRECtrl.updateEntity()
        print("Finish Run OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {} ".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "CreatDCEOPS":
        opsVECtrl , opsRECtrl , opsInfo = makeOPSInfoByCreatDCEOPS(runType, product, project, opsVersion, opsOrderJson, parameterJson, resultJson)
        print("Finish Creat DCE OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "RunDCEOPS":
        opsVECtrl , opsRECtrl , opsInfo = makeOPSInfoByRunDCEOPS(runType, product, project, opsVersion, opsRecordId)
        opsCtrl.executeDCE(opsInfo)
        opsRECtrl.setColumnValue("state", "FINISH")
        opsRECtrl.setColumnValue("resultjson", json.dumps(opsInfo["ResultJson"],ensure_ascii=False) if "ResultJson" in opsInfo.keys() else '{}')
        opsRECtrl.updateEntity()
        print("Finish Run DCE OPS , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    elif runType == "RunOnlyFunc":
        opsVECtrl , opsRECtrl , opsInfo = makeOPSInfoByRunOnlyFunc(runType, product, project, opsVersion, opsRecordId)
        opsInfo["OPSOrderJson"]["RepOPSRecordId"] = opsRecordId
        opsInfo["OPSOrderJson"]["RunFunctionArr"] = runFunctionArr
        opsCtrl.executeOPS(opsInfo)
        print("Finish Run Only Func , Product is {} , Project is {} , Version is {} , OPSRecordID is {}".format(opsInfo["Product"],opsInfo["Project"],opsInfo["OPSVersion"],opsInfo["OPSRecordId"]))
    return opsInfo

def makeOPSInfoByBuildOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    opsVersionEntityCtrl = OPSVersionEntity()
    opsRecordEntityCtrl = OPSRecordEntity()
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
    return opsVersionEntityCtrl , opsRecordEntityCtrl , opsInfo

def makeOPSInfoByRunOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    opsVersionEntityCtrl = OPSVersionEntity()
    opsRecordEntityCtrl = OPSRecordEntity()
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
    return opsVersionEntityCtrl , opsRecordEntityCtrl , opsInfo

def makeOPSInfoByCreatDCEOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson) :
    # 與makeOPSInfoByRunOPS模式一樣，只建立opsRecordEntity但不執行
    return makeOPSInfoByRunOPS (runType, product, project, opsVersion,opsOrderJson, parameterJson, resultJson)

def makeOPSInfoByRunDCEOPS (runType, product, project, opsVersion,opsRecordId) :
    opsVersionEntityCtrl = OPSVersionEntity()
    opsRecordEntityCtrl = OPSRecordEntity()
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
    return opsVersionEntityCtrl , opsRecordEntityCtrl , opsInfo

def makeOPSInfoByRunOnlyFunc (runType, product, project, opsVersion,opsRecordId) :
    # 與makeOPSInfoByRunDCEOPS模式一樣，只跑單一方法
    return makeOPSInfoByRunDCEOPS (runType, product, project, opsVersion,opsRecordId)
def makeOPSPipelinePNG(opsInfo,bgcolor="#FFFFFF",versionTextCount=10,functionTextCount=4):
    import pygraphviz
    filePath = "{}/{}/file/PLD".format(opsInfo["Product"][0], opsInfo["Project"][0])
    fileName = "{}_{}_{}.png".format(opsInfo["Product"][0], opsInfo["Project"][0], opsInfo["OPSVersion"][0])
    opsGraph = pygraphviz.AGraph(directed=True, strict=True, rankdir="LR", bgcolor=bgcolor)
    nodeLabel = opsInfo["OPSVersion"][0]
    if opsInfo["OPSVersion"][0] in opsInfo["OPSOrderJson"]["FunctionMemo"].keys():
        nodeLabel = opsInfo["OPSVersion"][0] + '\n' + opsInfo["OPSOrderJson"]["FunctionMemo"][opsInfo["OPSVersion"][0]][:versionTextCount]
    opsGraph.add_node(opsInfo["OPSVersion"][0], label=nodeLabel, fontname="SimHei", shape="square", style="diagonals")
    for exeFunction in opsInfo["OPSOrderJson"]["ExeFunctionArr"]:
        nodeLabel = exeFunction
        if exeFunction in opsInfo["OPSOrderJson"]["FunctionMemo"].keys():
            nodeLabel = exeFunction + '\n' + opsInfo["OPSOrderJson"]["FunctionMemo"][exeFunction][:functionTextCount]
        opsGraph.add_node(exeFunction, label=nodeLabel, fontname="SimHei", shape="square", style="solid")
        isHaveParent = False
        for ordFunction in opsInfo["OPSOrderJson"]["OrdFunctionArr"]:
            isHaveParent = True if ordFunction["Child"] == exeFunction else isHaveParent
        if isHaveParent == False:
            opsGraph.add_edge(opsInfo["OPSVersion"][0], exeFunction)
    for ordFunction in opsInfo["OPSOrderJson"]["OrdFunctionArr"]:
        opsGraph.add_edge(ordFunction["Parent"], ordFunction["Child"])
    opsGraph.graph_attr["epsilon"] = "0.001"
    opsGraph.layout("dot")
    os.makedirs(filePath) if not os.path.isdir(filePath) else None
    opsGraph.draw("{}/{}".format(filePath, fileName))



if __name__ == "__main__":
    main()