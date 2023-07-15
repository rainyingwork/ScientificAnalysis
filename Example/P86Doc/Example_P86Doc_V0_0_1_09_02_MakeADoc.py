import os
import Config
from Example.P86Doc.info.InfoMain import InfoMain
from package.artificialintelligence.common.common.DocumentCtrl import DocumentCtrl as AnalysisDocumentCtrl

analysisDocumentCtrl = AnalysisDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    makeInfo = {"DataTime":"2023-01-01"}

    dataMap = {
        "RawData": [
            infoMain.getInfo_R0_0_1(makeInfo)
        ],
    }
    fileName = "Example_P86Doc_AnalysisDoc.xlsx"
    initFilePath = 'common/common/file/doc/AnalysisDataDocInit.xlsx'
    outFilePath = 'Example/P86Doc/file/doc'
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    analysisDocumentCtrl.makeAnalysisDoc(dataMap,initFilePath,'{}/{}'.format(outFilePath,fileName))



