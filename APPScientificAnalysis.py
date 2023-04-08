import os, json; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy
from flask import Flask, request
from package.opsmanagement.common.LWLCtrl import LWLCtrl
from package.common.common.osbasic.BaseFunction import timethis

app = Flask(__name__)
lwlCtrl = LWLCtrl()

@app.route("/Example/P34PyTorch/V0_0_1", methods=['GET'])
@timethis
def Example_P34PyTorch_V0_0_1():
    # http://127.0.0.1:5000/Example/P34PyTorch/V0_0_1
    opsInfo = getJsonExample_P34PyTorch_V0_0_1()
    # opsInfo["ParameterJson"]["UP0_0_10"] = {"MovieName": request.values.get('MovieName')}
    lwlCtrl.executeRunFunction(opsInfo)
    return json.dumps(opsInfo["ResultJson"],ensure_ascii=False)

def getJsonExample_P34PyTorch_V0_0_1():
    basicInfo = {
        "RunType": ["RunOPS"]
        , "Product": ["Example"]
        , "Project": ["P34PyTorch"]
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": [
            "P0_0_10", "M0_0_10", "UP0_0_10",
        ],
        "RepOPSRecordId": 5,
        "RepFunctionArr": ["P0_0_10", "M0_0_10",],
        "RunFunctionArr": ["UP0_0_10",],  #
        "OrdFunctionArr": [
            {"Parent": "P0_0_10", "Child": "M0_0_10"}, {"Parent": "M0_0_10", "Child": "UP0_0_10"},
        ],
        "FunctionMemo": {
            "M0_0_10": "使用CNN進行圖片分類",
        },
    }
    opsInfo["ParameterJson"] = {
        "P0_0_10": {}, "M0_0_10": {}, "UP0_0_10": {},
    }
    opsInfo["ResultJson"] = {
    }
    return opsInfo

lwlCtrl.executePreviewReading(getJsonExample_P34PyTorch_V0_0_1())
app.run()


