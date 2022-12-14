from Example.P29RawData.circuit.P21Crawler import Crawler
from Example.P29RawData.circuit.P22Original import Original
from Example.P29RawData.circuit.P23Standard import Standard
from Example.P29RawData.circuit.P31RawData import RawData
from Example.P29RawData.circuit.P32PreProcess import PreProcess
from Example.P29RawData.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

