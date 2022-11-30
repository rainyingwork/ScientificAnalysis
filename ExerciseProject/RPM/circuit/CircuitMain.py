from ExerciseProject.RPM.circuit.P01Crawler import Crawler
from ExerciseProject.RPM.circuit.P02Original import Original
from ExerciseProject.RPM.circuit.P03Standard import Standard
from ExerciseProject.RPM.circuit.P31RawData import RawData
from ExerciseProject.RPM.circuit.P32PreProcess import PreProcess
from ExerciseProject.RPM.circuit.P33ModelUse import ModelUse
from ExerciseProject.RPM.circuit.P41ChartReport import ChartReport


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

