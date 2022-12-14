from Example.P35NetworkX.circuit.P21Crawler import Crawler
from Example.P35NetworkX.circuit.P22Original import Original
from Example.P35NetworkX.circuit.P23Standard import Standard
from Example.P35NetworkX.circuit.P31RawData import RawData
from Example.P35NetworkX.circuit.P32PreProcess import PreProcess
from Example.P35NetworkX.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

