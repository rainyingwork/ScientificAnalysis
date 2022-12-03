import os ,sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
from Example.P01Basic.info.InfoMain import InfoMain
from package.dataengineer.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

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

    fileName = "Example_P01Basic_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'Example/P01Basic/file/doc/{}'.format(fileName)
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,outFilePath)



