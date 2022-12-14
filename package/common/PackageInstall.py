import os , shutil
import sys

pythonEXEPath = "bin" if os.name == "posix" else "Scripts"

__pythonexePath = "{}/venv/{}/python".format(sys.path[1],pythonEXEPath)
# os.system("{} -m pip install pip==22.2.2".format(__pythonexePath))
# os.system("{} -m pip install setuptools==63.2.0".format(__pythonexePath))
# os.system("{} -m pip install python-dotenv==0.21.0".format(__pythonexePath))
# os.system("{} -m pip install gitpython==3.1.27".format(__pythonexePath))
# os.system("{} -m pip install paramiko==2.11.0".format(__pythonexePath))
# os.system("{} -m pip install datetime==4.5".format(__pythonexePath))
# os.system("{} -m pip install pandas==1.4.4".format(__pythonexePath))
# os.system("{} -m pip install numpy==1.23.5".format(__pythonexePath))
# os.system("{} -m pip install pyodbc==4.0.34".format(__pythonexePath))
# os.system("{} -m pip install SQLAlchemy==1.4.41".format(__pythonexePath))

# if os.name == "posix" :
#     os.system("{} -m pip install psycopg2-binary==2.9.3".format(__pythonexePath))
# else :
#     os.system("{} -m pip install psycopg2==2.9.3".format(__pythonexePath))

