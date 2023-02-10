from Example.P02DceOps.circuit.P21Crawler import Crawler
from Example.P02DceOps.circuit.P22Original import Original
from Example.P02DceOps.circuit.P23Standard import Standard
from Example.P02DceOps.circuit.P31RawData import RawData
from Example.P02DceOps.circuit.P32PreProcess import PreProcess
from Example.P02DceOps.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

