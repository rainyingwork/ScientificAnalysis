from Example.P15PreProcess.circuit.P21Crawler import Crawler
from Example.P15PreProcess.circuit.P22Original import Original
from Example.P15PreProcess.circuit.P23Standard import Standard
from Example.P15PreProcess.circuit.P31RawData import RawData
from Example.P15PreProcess.circuit.P32PreProcess import PreProcess
from Example.P15PreProcess.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

