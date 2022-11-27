from Example.P02Reduction.circuit.P01Crawler import Crawler
from Example.P02Reduction.circuit.P02Original import Original
from Example.P02Reduction.circuit.P03Standard import Standard
from Example.P02Reduction.circuit.P31RawData import RawData
from Example.P02Reduction.circuit.P32PreProcess import PreProcess
from Example.P02Reduction.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

