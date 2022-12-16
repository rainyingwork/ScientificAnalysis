from Example.P02DECOPS.circuit.P21Crawler import Crawler
from Example.P02DECOPS.circuit.P22Original import Original
from Example.P02DECOPS.circuit.P23Standard import Standard
from Example.P02DECOPS.circuit.P31RawData import RawData
from Example.P02DECOPS.circuit.P32PreProcess import PreProcess
from Example.P02DECOPS.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

