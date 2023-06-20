import os, json; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
import copy
from fastapi import FastAPI, Request
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.utils import get_openapi
from package.opsmanagement.common.LWLCtrl import LWLCtrl
from package.common.common.osbasic.BaseFunction import timethis

# http://127.0.0.1:5069/docs 文件網址

lwlCtrl = LWLCtrl()
app = FastAPI()

templates = Jinja2Templates(directory="Example/P69FastAPI/file/HTML")

@app.get("/Example/P69FastAPI/V0_0_1/{data}",
    summary="P69FastAPI V0_0_1",
    description="測試文件說明與網址",
)
async def Example_P69FastAPI_V0_0_1(request: Request, data: str):
    return templates.TemplateResponse("index.html", {"request": request, "data": data})

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5069)