from RWTaiwan.TagData.circuit.P21Crawler import Crawler
from RWTaiwan.TagData.circuit.P22Original import Original
from RWTaiwan.TagData.circuit.P23Standard import Standard
from RWTaiwan.TagData.circuit.P31RawData import RawData
from RWTaiwan.TagData.circuit.P32PreProcess import PreProcess
from RWTaiwan.TagData.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

