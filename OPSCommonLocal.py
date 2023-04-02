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