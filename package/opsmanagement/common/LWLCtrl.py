import pickle

class LWLCtrl:

    def __init__(self):
        self.allPGOD = {}

    def executePreviewReading(self, opsInfo):
        allGlobalObjectDict = {}
        opsInfo["GlobalObject"] = id(allGlobalObjectDict)
        product = opsInfo["Product"][0]
        project = opsInfo["Project"][0]
        opsVersion = opsInfo["OPSVersion"][0]
        repOPSRecordId = opsInfo["OPSOrderJson"]["RepOPSRecordId"]
        self.allPGOD[product] = {} if product not in self.allPGOD.keys() else self.allPGOD[product]
        self.allPGOD[product][project] = {} if project not in self.allPGOD[product].keys() else self.allPGOD[product][project]
        self.allPGOD[product][project][opsVersion] = {} if opsVersion not in self.allPGOD[product][project].keys() else self.allPGOD[product][project][opsVersion]
        self.allPGOD[product][project][opsVersion][repOPSRecordId] = allGlobalObjectDict if repOPSRecordId not in self.allPGOD[product][project][opsVersion].keys() else self.allPGOD[product][project][opsVersion][repOPSRecordId]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
            if executeFunction in opsInfo["OPSOrderJson"]["RepFunctionArr"]:
                exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
                with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
                    opsInfo["ResultJson"][executeFunction] = pickle.load(fr)
                with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
                    allGlobalObjectDict[executeFunction] = pickle.load(god)

    def executeAllFunction(self, opsInfo):
        allGlobalObjectDict = {}
        opsInfo["GlobalObject"] = id(allGlobalObjectDict)
        product = opsInfo["Product"][0]
        project = opsInfo["Project"][0]
        opsVersion = opsInfo["OPSVersion"][0]
        repOPSRecordId = opsInfo["OPSOrderJson"]["RepOPSRecordId"]
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
            if executeFunction in opsInfo["OPSOrderJson"]["RepFunctionArr"]:
                exeFunctionLDir = "{}/{}/file/result/{}/{}/{}".format(product, project, opsVersion, str(repOPSRecordId),executeFunction)
                with open('{}/{}'.format(exeFunctionLDir, '/FunctionRestlt.pickle'), 'rb') as fr:
                    opsInfo["ResultJson"][executeFunction] = pickle.load(fr)
                with open('{}/{}'.format(exeFunctionLDir, '/GlobalObjectDict.pickle'), 'rb') as god:
                    allGlobalObjectDict[executeFunction] = pickle.load(god)
            if executeFunction in opsInfo["OPSOrderJson"]["RunFunctionArr"]:
                opsInfo["ResultJson"][executeFunction], allGlobalObjectDict[executeFunction] = eval(f"circuitMain.{executeFunction}({opsInfo})")
        return opsInfo

    def executeRunFunction(self, opsInfo):
        product = opsInfo["Product"][0]
        project = opsInfo["Project"][0]
        opsVersion = opsInfo["OPSVersion"][0]
        repOPSRecordId = opsInfo["OPSOrderJson"]["RepOPSRecordId"]
        allGlobalObjectDict = self.allPGOD[product][project][opsVersion][repOPSRecordId]
        opsInfo["GlobalObject"] = id(allGlobalObjectDict)
        eval(f"exec('from {product}.{project}.circuit.CircuitMain import CircuitMain')")
        circuitMain = eval(f"CircuitMain()")
        for executeFunction in opsInfo["OPSOrderJson"]["OrderFunctions"]:
            if executeFunction in opsInfo["OPSOrderJson"]["RunFunctionArr"]:
                opsInfo["ResultJson"][executeFunction], allGlobalObjectDict[executeFunction] = eval(f"circuitMain.{executeFunction}({opsInfo})")
        return opsInfo