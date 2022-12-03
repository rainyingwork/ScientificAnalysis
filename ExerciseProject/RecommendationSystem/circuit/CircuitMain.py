from ExerciseProject.RecommendationSystem.circuit.P01Crawler import Crawler
from ExerciseProject.RecommendationSystem.circuit.P02Original import Original
from ExerciseProject.RecommendationSystem.circuit.P03Standard import Standard
from ExerciseProject.RecommendationSystem.circuit.P31RawData import RawData
from ExerciseProject.RecommendationSystem.circuit.P32PreProcess import PreProcess
from ExerciseProject.RecommendationSystem.circuit.P33ModelUse import ModelUse

class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    ) :
    pass

