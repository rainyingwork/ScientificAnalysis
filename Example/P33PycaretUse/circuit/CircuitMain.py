from Example.P33PycaretUse.circuit.P21Crawler import Crawler
from Example.P33PycaretUse.circuit.P22Original import Original
from Example.P33PycaretUse.circuit.P23Standard import Standard
from Example.P33PycaretUse.circuit.P31RawData import RawData
from Example.P33PycaretUse.circuit.P32PreProcess import PreProcess
from Example.P33PycaretUse.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

