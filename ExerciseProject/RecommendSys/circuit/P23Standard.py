from ExerciseProject.RecommendSys.circuit.CorrelationAnalysis.P23Standard import Standard as Standard_CorrelationAnalysis
from ExerciseProject.RecommendSys.circuit.ContentBasedFiltering.P23Standard import Standard as Standard_ContentBasedFiltering
from ExerciseProject.RecommendSys.circuit.CollaborativeFiltering.P23Standard import Standard as Standard_CollaborativeFiltering

class Standard (
    Standard_CorrelationAnalysis
    , Standard_ContentBasedFiltering
    , Standard_CollaborativeFiltering
    ) :
    pass
