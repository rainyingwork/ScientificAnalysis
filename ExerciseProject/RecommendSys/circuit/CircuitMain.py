from ExerciseProject.RecommendSys.circuit.P01Crawler import Crawler
from ExerciseProject.RecommendSys.circuit.P02Original import Original
from ExerciseProject.RecommendSys.circuit.P03Standard import Standard
from ExerciseProject.RecommendSys.circuit.P31RawData import RawData
from ExerciseProject.RecommendSys.circuit.P32PreProcess import PreProcess
from ExerciseProject.RecommendSys.circuit.P33ModelUse import ModelUse
from ExerciseProject.RecommendSys.circuit.P41UseProduct import UseProduct

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    , UseProduct
    ) :
    pass

