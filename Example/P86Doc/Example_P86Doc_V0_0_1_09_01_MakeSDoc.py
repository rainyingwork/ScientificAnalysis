import os
from Example.P86Doc.info.InfoMain import InfoMain
from package.dataengineer.common.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

standardDocumentCtrl = StandardDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    dataMap = {
        "Standard" : [
            infoMain.getInfo_S0_0_1(),
        ],
    }
    fileName = "Example_P86Doc_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'Example/P86Doc/file/doc'
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,'{}/{}'.format(outFilePath,fileName))



