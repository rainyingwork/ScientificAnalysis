import os; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
from Example.P01Basic.info.InfoMain import InfoMain
from package.dataengineer.common.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

standardDocumentCtrl = StandardDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    dataMap = {
        "AllData" : []
        , "Crawler" : []
        , "Original" : []
        , "Standard" : []
        , "RawData": []
        , "PreProcess": []
        , "ModelUse": []
    }
    os.makedirs('ExerciseProject/RPM/file/doc/') if not os.path.isdir('ExerciseProject/RPM/file/doc/') else None
    fileName = "ExerciseProject_RPM_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'ExerciseProject/RPM/file/doc/{}'.format(fileName)
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,outFilePath)



