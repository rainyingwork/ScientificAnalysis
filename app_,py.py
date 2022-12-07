import os; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy , pickle , json
from package.common.osbasic.BaseFunction import timethis
from flask import Flask, request

app = Flask(__name__)

@app.route("/ExerciseProject/RecommendSys/V0_0_2", methods=['GET'])
@timethis
def Maple_V0_0_2():
    opsInfo = {
        "RunType": ["runops"]
        , "Product": ["ExerciseProject"]
        , "Project": ["RecommendSys"]
    }
    opsInfo["OPSVersion"] = ["V0_1_2"]
    opsInfo["OPSOrderJson"] = {
        "OrderFunctions": ["M0_1_2", "UP0_1_1"]
        , "RepOPSRecordId": 1214
        , "RepFunctionArr": ["M0_1_2"]
        , "RunFunctionArr": ["UP0_1_1"]
    }
    opsInfo["ParameterJson"] = {
        "UP0_1_1": {"MovieName": request.values.get('MovieName')}
    }
    opsInfo["ResultJson"] = {}
    opsInfo = runOPS(opsInfo)
    return json.dumps(opsInfo["ResultJson"],ensure_ascii=False)

def runOPS(opsInfo):
    allGlobalObjectDict = {}
    opsInfo["GlobalObject"] = id(allGlobalObjectDict)
    product = opsInfo["Product"][0]
    project = opsInfo["Project"][0]
    opsVersion = opsInfo["OPSVersion"][0]
    repOPSRecordId = opsInfo["OPSOrderJson"]["RepOPSRecordId"]
    eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
    circuitMain = eval(f"CircuitMain()")
    for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
        if executeFunction in opsInfo["OPSOrderJson"]["RepFunctionArr"]:
            exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
            with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
                opsInfo["ResultJson"][executeFunction] = pickle.load(fr)
            with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
                allGlobalObjectDict[executeFunction] = pickle.load(god)
        if executeFunction in opsInfo["OPSOrderJson"]["RunFunctionArr"]:
            opsInfo["ResultJson"][executeFunction], allGlobalObjectDict[executeFunction] = eval(
                f"circuitMain.{executeFunction}({opsInfo})")
    return opsInfo

app.run()