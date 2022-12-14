from Example.P14RawData.circuit.P21Crawler import Crawler
from Example.P14RawData.circuit.P22Original import Original
from Example.P14RawData.circuit.P23Standard import Standard
from Example.P14RawData.circuit.P31RawData import RawData
from Example.P14RawData.circuit.P32PreProcess import PreProcess
from Example.P14RawData.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

