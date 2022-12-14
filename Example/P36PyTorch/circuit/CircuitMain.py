from Example.P36PyTorch.circuit.P21Crawler import Crawler
from Example.P36PyTorch.circuit.P22Original import Original
from Example.P36PyTorch.circuit.P23Standard import Standard
from Example.P36PyTorch.circuit.P31RawData import RawData
from Example.P36PyTorch.circuit.P32PreProcess import PreProcess
from Example.P36PyTorch.circuit.P33ModelUse import ModelUse
from Example.P36PyTorch.circuit.P41ChartReport import ChartReport
from Example.P36PyTorch.circuit.P41UseProduct import UseProduct

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    , ChartReport
    , UseProduct
    ) :
    pass

