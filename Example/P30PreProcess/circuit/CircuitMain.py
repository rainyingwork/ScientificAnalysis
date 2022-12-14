from Example.P30PreProcess.circuit.P21Crawler import Crawler
from Example.P30PreProcess.circuit.P22Original import Original
from Example.P30PreProcess.circuit.P23Standard import Standard
from Example.P30PreProcess.circuit.P31RawData import RawData
from Example.P30PreProcess.circuit.P32PreProcess import PreProcess
from Example.P30PreProcess.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

