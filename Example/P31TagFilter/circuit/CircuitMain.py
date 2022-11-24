from Example.P31TagFilter.circuit.P01Crawler import Crawler
from Example.P31TagFilter.circuit.P02Original import Original
from Example.P31TagFilter.circuit.P03Standard import Standard
from Example.P31TagFilter.circuit.P31RawData import RawData
from Example.P31TagFilter.circuit.P32PreProcess import PreProcess
from Example.P31TagFilter.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

