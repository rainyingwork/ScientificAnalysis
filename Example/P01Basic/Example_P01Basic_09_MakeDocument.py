import os
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
    fileName = "Example_P01Basic_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'Example/P01Basic/file/doc'
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,'{}/{}'.format(outFilePath,fileName))



