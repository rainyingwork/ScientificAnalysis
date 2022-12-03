import os ,sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
from ExerciseProject.RecommendSys.info.InfoMain import InfoMain
from package.dataengineer.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

standardDocumentCtrl = StandardDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    dataMap = {
        "Standard" : [infoMain.getInfo_S0_0_1()]
    }
    fileName = "ExerciseProject_RecommendSys_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'ExerciseProject/RecommendSys/file/doc/{}'.format(fileName)
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,'{}/{}'.format(outFilePath,fileName))



