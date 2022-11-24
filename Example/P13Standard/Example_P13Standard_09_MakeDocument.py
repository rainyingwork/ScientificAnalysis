import os ,sys ; os.chdir(os.path.dirname(__file__)) if os.name == "posix" else None
from Example.P13Standard.info.InfoMain import InfoMain
from package.dataengineer.standard.DocumentCtrl import DocumentCtrl as StandardDocumentCtrl

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

    fileName = "Example_P13Standard_StandardDoc.xlsx"
    initFilePath = 'common/common/file/doc/StandardDataDocInit.xlsx'
    outFilePath = 'Example/P13Standard/file/doc/{}'.format(fileName)
    standardDocumentCtrl.MakeStandardDoc(dataMap,initFilePath,outFilePath)

    # fileName = "ExampleProduct_P03DocumentExample_ModelUseDataOmitDoc.xlsx"
    # initFilePath = 'common/common/file/doc/AnalysisDataOmitDocInit.xlsx'
    # outFilePath = 'ExampleProduct/P03DocumentExample/file/doc/{}'.format(fileName)
    # standardDocumentCtrl.MakeModelUseDataOmitDoc(dataMap, initFilePath, outFilePath)


