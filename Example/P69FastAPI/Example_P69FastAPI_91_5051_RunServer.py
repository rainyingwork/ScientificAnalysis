import os, json; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy
from fastapi import FastAPI, Request
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from package.opsmanagement.common.LWLCtrl import LWLCtrl
from package.common.common.osbasic.BaseFunction import timethis

lwlCtrl = LWLCtrl()
app = FastAPI()

def getJsonExample_P51OLAP_V0_0_1():
    basicInfo = {
        "RunType": ["RunOPS"],
        "Product": ["Example"],
        "Project": ["P51OLAP"],
    }
    opsInfo = copy.deepcopy(basicInfo)
    opsInfo["OPSVersion"] = ["V0_0_1"]
    # opsInfo["OPSRecordId"] = []
    opsInfo["OPSOrderJson"] = {
        "ExeFunctionArr": ["R0_0_1", "UP0_0_1"],
        "RepOPSRecordId": 2701,
        "RepFunctionArr": ["R0_0_1"],
        "RunFunctionArr": ["UP0_0_1"],
        "OrdFunctionArr": [
            {"Parent": "R0_0_1", "Child": "UP0_0_1"},
        ],
        "FunctionMemo": {
            "R0_0_1": "",
            "UP0_0_1": "",
        },
    }
    opsInfo["ParameterJson"] = {
        "R0_0_1": {},
        "UP0_0_1": {},
    }
    opsInfo["ResultJson"] = {}
    return opsInfo

@app.get("/Example/P51OLAP/V0_0_1")
async def Example_P51OLAP_V0_0_1():
    opsInfo = getJsonExample_P51OLAP_V0_0_1()
    lwlCtrl.reloadCircuit(opsInfo)
    lwlCtrl.executeRunFunction(opsInfo)
    return opsInfo["ResultJson"]

if __name__ == '__main__':
    lwlCtrl.executePreviewReading(getJsonExample_P51OLAP_V0_0_1())
    uvicorn.run(app, host="127.0.0.1", port=5051)