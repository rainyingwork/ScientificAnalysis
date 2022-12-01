from ExerciseProject.CustomerRetention.circuit.P01Crawler import Crawler
from ExerciseProject.CustomerRetention.circuit.P02Original import Original
from ExerciseProject.CustomerRetention.circuit.P03Standard import Standard
from ExerciseProject.CustomerRetention.circuit.P31RawData import RawData
from ExerciseProject.CustomerRetention.circuit.P32PreProcess import PreProcess
from ExerciseProject.CustomerRetention.circuit.P33ModelUse import ModelUse
from ExerciseProject.CustomerRetention.circuit.P41ChartReport import ChartReport


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

