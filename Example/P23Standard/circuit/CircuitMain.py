from Example.P23Standard.circuit.P21Crawler import Crawler
from Example.P23Standard.circuit.P22Original import Original
from Example.P23Standard.circuit.P23Standard import Standard
from Example.P23Standard.circuit.P31RawData import RawData
from Example.P23Standard.circuit.P32PreProcess import PreProcess
from Example.P23Standard.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

