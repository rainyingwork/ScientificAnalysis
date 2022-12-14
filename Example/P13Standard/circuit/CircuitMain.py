from Example.P13Standard.circuit.P21Crawler import Crawler
from Example.P13Standard.circuit.P22Original import Original
from Example.P13Standard.circuit.P23Standard import Standard
from Example.P13Standard.circuit.P31RawData import RawData
from Example.P13Standard.circuit.P32PreProcess import PreProcess
from Example.P13Standard.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

