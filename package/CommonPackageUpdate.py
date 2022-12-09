import os , shutil
import sys

pythonEXEPath = "bin" if os.name == "posix" else "Scripts"

__pythonexePath = "{}/venv/{}/python".format(sys.path[1],pythonEXEPath)
os.system("{} -m pip install scikit-learn==1.1.2".format(__pythonexePath))


# os.system("{} -m pip install --upgrade pip".format(__pythonexePath))
# os.system("{} -m pip install gitpython".format(__pythonexePath))
# os.system("{} -m pip install matplotlib".format(__pythonexePath))
# os.system("{} -m pip install scikit-learn".format(__pythonexePath))
# os.system("{} -m pip install nltk".format(__pythonexePath))
# os.system("{} -m pip install streamlit".format(__pythonexePath))
# os.system("{} -m pip install scikit-surprise".format(__pythonexePath))
# os.system("{} -m pip install seaborn".format(__pythonexePath))
# os.system("{} -m pip install pycaret==3.0.0rc2".format(__pythonexePath))
# os.system("{} -m pip install xgboost==1.6.2".format(__pythonexePath))
# os.system("{} -m pip install catboost==1.0.6".format(__pythonexePath))
# os.system("{} -m pip install gensim==4.2.0".format(__pythonexePath))
# os.system("{} -m pip install tqdm==4.64.1".format(__pythonexePath))
# os.system("{} -m pip install openpyxl==3.0.10".format(__pythonexePath))
# os.system("{} -m pip install matplotlib==3.6.0".format(__pythonexePath))
# os.system("{} -m pip install explainerdashboard==0.4.0".format(__pythonexePath))
# os.system("{} -m pip install networkx==2.8.8".format(__pythonexePath))
# os.system("{} -m pip install m2cgen==0.10.0".format(__pythonexePath))
# os.system("{} -m pip install evidently==0.2.0".format(__pythonexePath))
# os.system("{} -m pip install mlxtend==0.21.0".format(__pythonexePath))
# os.system("{} -m pip install rake_nltk==1.0.6".format(__pythonexePath))
# os.system("{} -m pip install Flask==2.2.2".format(__pythonexePath))
# os.system("{} -m pip install impyla==0.18.0".format(__pythonexePath))
# os.system("{} -m pip install hdfs==2.7.0".format(__pythonexePath))
#os.system("{} -m pip install pyspark==3.3.1".format(__pythonexePath))
# os.system("{} -m pip install py4j==0.10.9.7".format(__pythonexePath))
# os.system("{} -m pip install tensorflow==2.11.0".format(__pythonexePath))




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
