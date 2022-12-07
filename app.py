import os; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import json
from package.opsmanagement.LWLCtrl import LWLCtrl
from package.common.osbasic.BaseFunction import timethis
from flask import Flask, request

app = Flask(__name__)
lwlCtrl = LWLCtrl()

@app.route("/ExerciseProject/RecommendSys/V0_0_2", methods=['GET'])
@timethis
def RecommendSysV0_0_2():
    # /ExerciseProject/RecommendSys/V0_0_2?MovieName=Bride Wars
    opsInfo = getJsonRecommendSys_V0_0_2()
    opsInfo["ParameterJson"]["UP0_1_1"] = {"MovieName": request.values.get('MovieName')}
    lwlCtrl.executeRunFunction(opsInfo)
    return json.dumps(opsInfo["ResultJson"],ensure_ascii=False)

def getJsonRecommendSys_V0_0_2():
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
        "UP0_1_1": {}
    }
    opsInfo["ResultJson"] = {}
    return opsInfo

lwlCtrl.executePreviewReading(getJsonRecommendSys_V0_0_2())

app.run()


