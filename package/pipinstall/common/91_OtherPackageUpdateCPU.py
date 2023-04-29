import os , shutil
import sys

isInstall = True
isLocalDir = False
pythonEXEPath = "python" if os.path.isfile("venv/python.exe") else "bin/python" if os.name == "posix" else "Scripts/python"
__pythonexePath = "{}/venv/{}".format(sys.path[1],pythonEXEPath)
__pipFunction = ("pip install --no-index --find-links=venv/pip" if isLocalDir == True else "pip install") if isInstall == True else "pip download -d venv/pip"

# Other套件 -------------------------------------------------
# os.system("{} -m {} openai==0.27.2".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3
os.system("{} -m {} gTTS==2.3.2".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3

# 安裝pygraphviz 請先至 graphviz 安裝2.46以上版本下載位置 https://graphviz.org/download/
# os.system('{} -m {} --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz==1.10'.format(__pythonexePath,__pipFunction))

# os.system("{} -m {} diffusers==0.14.0".format(__pythonexePath,__pipFunction))          # diffusers
# os.system("{} -m {} transformers==4.27.4".format(__pythonexePath,__pipFunction))          # diffusers

# os.system("{} -m {} pivottablejs==0.9.0".format(__pythonexePath,__pipFunction))          # OpenAI GPT-3