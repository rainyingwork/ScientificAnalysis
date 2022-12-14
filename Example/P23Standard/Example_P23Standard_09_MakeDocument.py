import os; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
from Example.P23Standard.info.InfoMain import InfoMain
from package.dataengineer.common.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

standardDocumentCtrl = StandardDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    dataMap = {
        "AllData" : [
            infoMain.getInfo_S0_0_1()
        ]
        , "Crawler" : []
        , "Original" : []
        , "Standard" : [
            infoMain.getInfo_S0_0_1()
        ]
        , "RawData": []
        , "PreProcess": []
        , "ModelUse": []
    }

    fileName = "Example_P23Standard_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'Example/P23Standard/file/doc/'
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    standardDocumentCtrl.MakeStandardDoc(dataMap, initFilePath, '{}/{}'.format(outFilePath, fileName))



