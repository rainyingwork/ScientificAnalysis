from ExerciseProject.RFM.circuit.P01Crawler import Crawler
from ExerciseProject.RFM.circuit.P02Original import Original
from ExerciseProject.RFM.circuit.P03Standard import Standard
from ExerciseProject.RFM.circuit.P31RawData import RawData
from ExerciseProject.RFM.circuit.P32PreProcess import PreProcess
from ExerciseProject.RFM.circuit.P33ModelUse import ModelUse
from ExerciseProject.RFM.circuit.P41ChartReport import ChartReport


class CircuitMain(
    Crawler
    , Original
    , Standard
    , RawData
    , PreProcess
    , ModelUse
    , ChartReport
    ) :
    pass

