from Project.RecommendationSystem.circuit.P01Crawler import Crawler
from Project.RecommendationSystem.circuit.P02Original import Original
from Project.RecommendationSystem.circuit.P03Standard import Standard
from Project.RecommendationSystem.circuit.P31RawData import RawData
from Project.RecommendationSystem.circuit.P32PreProcess import PreProcess
from Project.RecommendationSystem.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

