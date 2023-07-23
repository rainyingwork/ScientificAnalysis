import os
from Example.P29RawData.info.InfoMain import InfoMain
from package.artificialintelligence.common.common.DocumentCtrl import DocumentCtrl as AnalysisDocumentCtrl
import Config

analysisDocumentCtrl = AnalysisDocumentCtrl()
infoMain = InfoMain()

if __name__ == "__main__":
    dataMap = {
        "AllData" : [
            infoMain.getInfo_R0_0_1_ByFile(),
        ],
        "Crawler" : [],
        "Original" : [],
        "Standard" : [
        ],
        "RawData": [
            infoMain.getInfo_R0_0_1_ByDB(),
        ],
        "PreProcess": [],
        "ModelUse": [],
    }

    fileName = "Example_P29RawData_AnalysisDoc.xlsx"
    initFilePath = 'common/common/file/doc/AnalysisDataDocInit.xlsx'
    outFilePath = 'Example/P29RawData/file/doc/'
    os.makedirs(outFilePath) if not os.path.isdir(outFilePath) else None
    analysisDocumentCtrl.makeAnalysisDoc(dataMap, initFilePath, '{}/{}'.format(outFilePath, fileName))





