from UnitTest.TensorFlow.circuit.P01Crawler import Crawler
from UnitTest.TensorFlow.circuit.P02Original import Original
from UnitTest.TensorFlow.circuit.P03Standard import Standard
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

