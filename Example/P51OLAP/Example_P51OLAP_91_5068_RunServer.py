import os, json; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy
from fastapi import FastAPI, Request
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from package.opsmanagement.common.LWLCtrl import LWLCtrl
from package.common.common.osbasic.BaseFunction import timethis
from flask import Flask, render_template

lwlCtrl = LWLCtrl()

app = Flask(__name__, template_folder='file/html' , static_folder='file/html')

@app.route("/Example/P51OLAP/O1_0_1")
def Example_P51OLAP_O1_0_1():
    opsInfo = getJsonExample_P51OLAP_V0_0_1()
    opsInfo["OPSOrderJson"]["ExeFunctionArr"] = ["O1_0_1"]
    lwlCtrl.reloadCircuit(opsInfo)
    lwlCtrl.executeRunFunction(opsInfo)
    return opsInfo["ResultJson"]

@app.route("/Example/P51OLAP/O1_0_2")
def Example_P51OLAP_O1_0_2():
    opsInfo = getJsonExample_P51OLAP_V0_0_1()
    opsInfo["OPSOrderJson"]["ExeFunctionArr"] = ["O1_0_2"]
    lwlCtrl.reloadCircuit(opsInfo)
    lwlCtrl.executeRunFunction(opsInfo)
    return opsInfo["ResultJson"]

def getJsonExample_P51OLAP_V0_0_1():
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P51OLAP"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    opsInfo["OPSRecordId"] = ["9999"]
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": [],
        "RepOPSRecordId": 9999,
        "RepFunctionArr": [],
        "RunFunctionArr": ["O1_0_1","O1_0_2"],
        "OrdFunctionArr": [],
        "FunctionMemo": {},
    }
    opsInfo["ParameterJson"] = {
    }
    opsInfo["ResultJson"] = {}
    return opsInfo

if __name__ == '__main__':
    lwlCtrl.executePreviewReading(getJsonExample_P51OLAP_V0_0_1())
    app.run(debug=False,port=5068)