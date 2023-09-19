from Example.P41AutoTrainCycle.circuit.P21Crawler import Crawler
from Example.P41AutoTrainCycle.circuit.P22Original import Original
from Example.P41AutoTrainCycle.circuit.P23Standard import Standard
from Example.P41AutoTrainCycle.circuit.P31RawData import RawData
from Example.P41AutoTrainCycle.circuit.P32PreProcess import PreProcess
from Example.P41AutoTrainCycle.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

