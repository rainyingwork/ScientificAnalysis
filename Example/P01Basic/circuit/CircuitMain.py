from Example.P01Basic.circuit.P12Docker import Docker
from Example.P01Basic.circuit.P21Crawler import Crawler
from Example.P01Basic.circuit.P22Original import Original
from Example.P01Basic.circuit.P23Standard import Standard
from Example.P01Basic.circuit.P31RawData import RawData
from Example.P01Basic.circuit.P32PreProcess import PreProcess
from Example.P01Basic.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Docker
    , Crawler, Original, Standard
    , RawData, PreProcess, ModelUse
    ) :
    pass

