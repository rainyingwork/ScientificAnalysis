import os; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy , pickle , json
from package.common.osbasic.BaseFunction import timethis
from flask import Flask, request

app = Flask(__name__)

@app.route("/ExerciseProject/RecommendSys/V0_0_2", methods=['GET'])
@timethis
def Maple_V0_0_2():
    # /ExerciseProject/RecommendSys/V0_0_2?MovieName=Bride Wars
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
    opsInfo["HTMLResultJson"] = {}
    opsInfo["GlobalObject"] = id(allGlobalObjectDict[product][project][opsVersion][repOPSRecordId])
    for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
        if executeFunction in opsInfo["OPSOrderJson"]["RunFunctionArr"]:
            opsInfo["HTMLResultJson"][executeFunction], allGlobalObjectDict[executeFunction] = eval(f"circuitMain.{executeFunction}({opsInfo})")
    return json.dumps(opsInfo["HTMLResultJson"],ensure_ascii=False)

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
opsInfo["ParameterJson"] = {}
opsInfo["ResultJson"] = {}

product , project , opsVersion , repOPSRecordId= opsInfo["Product"][0] , opsInfo["Project"][0] , opsInfo["OPSVersion"][0] , opsInfo["OPSOrderJson"]["RepOPSRecordId"]
eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
circuitMain = eval(f"CircuitMain()")

allGlobalObjectDict = {}
allGlobalObjectDict[product] = {}
allGlobalObjectDict[product][project] = {}
allGlobalObjectDict[product][project][opsVersion] = {}
allGlobalObjectDict[product][project][opsVersion][repOPSRecordId] = {}
for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
    if executeFunction in opsInfo["OPSOrderJson"]["RepFunctionArr"]:
        exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
        with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
            allGlobalObjectDict[product][project][opsVersion][repOPSRecordId][executeFunction] = pickle.load(god)

app.run()


