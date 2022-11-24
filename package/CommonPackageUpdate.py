import os , shutil
import sys

__pythonexePath = "{}/venv/Scripts/python.exe".format(sys.path[1])
os.system("{} -m pip install --upgrade pip".format(__pythonexePath))
os.system("{} -m pip install gitpython".format(__pythonexePath))
os.system("{} -m pip install matplotlib".format(__pythonexePath))
os.system("{} -m pip install scikit-learn".format(__pythonexePath))
os.system("{} -m pip install nltk".format(__pythonexePath))
os.system("{} -m pip install streamlit".format(__pythonexePath))
os.system("{} -m pip install scikit-surprise".format(__pythonexePath))
os.system("{} -m pip install seaborn".format(__pythonexePath))



# import git
# import errno
# import stat
#
# def handle_remove_read_only(func, path, exc):
#     excvalue = exc[1]
#     if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
#       os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
#       func(path)
#     else:
#       raise
#
# gitURL = "http://127.0.0.1:8888/work/PackageLibrary.git"
# sourceFolderRoot = "source"
# sourceFolderPath = sourceFolderRoot + "/package/common"
# targetFolderPath = "package/common"
#
# shutil.rmtree(sourceFolderRoot , ignore_errors=True)
# shutil.rmtree(targetFolderPath, ignore_errors=True)
# git.Repo.clone_from(url=gitURL, to_path=sourceFolderRoot)
# shutil.move(sourceFolderPath , targetFolderPath)
# shutil.rmtree(sourceFolderRoot, onerror=handle_remove_read_only)
