from ExerciseProject.RecommendSys.circuit.CorrelationAnalysis.P32PreProcess import PreProcess as PreProcess_CorrelationAnalysis
from ExerciseProject.RecommendSys.circuit.ContentBasedFiltering.P32PreProcess import PreProcess as PreProcess_ContentBasedFiltering
from ExerciseProject.RecommendSys.circuit.CollaborativeFiltering.P32PreProcess import PreProcess as PreProcess_CollaborativeFiltering

class PreProcess(
    PreProcess_CorrelationAnalysis
    , PreProcess_ContentBasedFiltering
    , PreProcess_CollaborativeFiltering
    ) :
    pass






