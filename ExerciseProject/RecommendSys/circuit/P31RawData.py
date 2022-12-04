from ExerciseProject.RecommendSys.circuit.CorrelationAnalysis.P31RawData import RawData as RawData_CorrelationAnalysis
from ExerciseProject.RecommendSys.circuit.ContentBasedFiltering.P31RawData import RawData as RawData_ContentBasedFiltering
from ExerciseProject.RecommendSys.circuit.CollaborativeFiltering.P31RawData import RawData as RawData_CollaborativeFiltering

class RawData(
    RawData_CorrelationAnalysis
    , RawData_ContentBasedFiltering
    , RawData_CollaborativeFiltering
    ) :
    pass



