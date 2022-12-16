from Example.P31ScikitLearn.circuit.P21Crawler import Crawler
from Example.P31ScikitLearn.circuit.P22Original import Original
from Example.P31ScikitLearn.circuit.P23Standard import Standard
from Example.P31ScikitLearn.circuit.P31RawData import RawData
from Example.P31ScikitLearn.circuit.P32PreProcess import PreProcess
from Example.P31ScikitLearn.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

