from UnitTest.PyTorch.circuit.P01Crawler import Crawler
from UnitTest.PyTorch.circuit.P02Original import Original
from UnitTest.PyTorch.circuit.P03Standard import Standard
from UnitTest.PyTorch.circuit.P31RawData import RawData
from UnitTest.PyTorch.circuit.P32PreProcess import PreProcess
from UnitTest.PyTorch.circuit.P33ModelUse import ModelUse
from UnitTest.PyTorch.circuit.P41ChartReport import ChartReport
from UnitTest.PyTorch.circuit.P41UseProduct import UseProduct

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

