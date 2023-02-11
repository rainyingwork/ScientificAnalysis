from Example.P03LoadBalance.circuit.P21Crawler import Crawler
from Example.P03LoadBalance.circuit.P22Original import Original
from Example.P03LoadBalance.circuit.P23Standard import Standard
from Example.P03LoadBalance.circuit.P31RawData import RawData
from Example.P03LoadBalance.circuit.P32PreProcess import PreProcess
from Example.P03LoadBalance.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

