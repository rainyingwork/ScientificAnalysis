import os , shutil
import sys

isInstall = False
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = "pip install --no-index --find-links=venv\pip" if isLocalDir == True else "pip install" if isInstall == True else "pip download -d venv\pip"

# Other套件 -------------------------------------------------
os.system("{} -m {} openai==0.25.0".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3
