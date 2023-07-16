import os , shutil
import sys

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = ("pip install --no-index --find-links=venv/pip" if isLocalDir == True else "pip install") if isInstall == True else "pip download -d venv/pip"

# Other套件 -------------------------------------------------
# os.system("{} -m {} pivottablejs==0.9.0".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3
# os.system("{} -m {} comtradeapicall==1.0.20".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3
