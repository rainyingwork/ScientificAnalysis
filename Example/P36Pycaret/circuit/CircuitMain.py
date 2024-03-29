from Example.P36Pycaret.circuit.P21Crawler import Crawler
from Example.P36Pycaret.circuit.P22Original import Original
from Example.P36Pycaret.circuit.P23Standard import Standard
from Example.P36Pycaret.circuit.P31RawData import RawData
from Example.P36Pycaret.circuit.P32PreProcess import PreProcess
from Example.P36Pycaret.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

