from Example.P01Basic.circuit.P01Crawler import Crawler
from Example.P01Basic.circuit.P02Original import Original
from Example.P01Basic.circuit.P03Standard import Standard
from Example.P01Basic.circuit.P31RawData import RawData
from Example.P01Basic.circuit.P32PreProcess import PreProcess
from Example.P01Basic.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

