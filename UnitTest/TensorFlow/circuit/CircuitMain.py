from UnitTest.TensorFlow.circuit.P21Crawler import Crawler
from UnitTest.TensorFlow.circuit.P22Original import Original
from UnitTest.TensorFlow.circuit.P23Standard import Standard
from UnitTest.TensorFlow.circuit.P31RawData import RawData
from UnitTest.TensorFlow.circuit.P32PreProcess import PreProcess
from UnitTest.TensorFlow.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

