import os , shutil
import sys

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = ("pip install --no-index --find-links=venv/pip" if isLocalDir == True else "pip install") if isInstall == True else "pip download -d venv/pip"

# Main 套件 -------------------------------------------------
# os.system("{} -m {} transformers==4.27.4".format(__pythonexePath,__pipFunction))          # diffusers
# OpenAI 套件 -------------------------------------------------
# os.system("{} -m {} openai==0.27.2".format(__pythonexePath,__pipFunction))
# StableDiffusion 套件 -------------------------------------------------
# os.system("{} -m {} diffusers==0.14.0".format(__pythonexePath,__pipFunction))             # diffusers
# GenVoice 套件 -------------------------------------------------
# os.system("{} -m {} gTTS==2.3.2".format(__pythonexePath,__pipFunction))                     # 聲音套件
# os.system("{} -m {} pydub==0.25.1".format(__pythonexePath,__pipFunction))                   # 聲音套件
# os.system("{} -m {} ffmpeg==1.4".format(__pythonexePath,__pipFunction))                   # 聲音套件
