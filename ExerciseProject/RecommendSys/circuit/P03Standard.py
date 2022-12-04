from ExerciseProject.RecommendSys.circuit.CorrelationAnalysis.P03Standard import Standard as Standard_CorrelationAnalysis
from ExerciseProject.RecommendSys.circuit.ContentBasedFiltering.P03Standard import Standard as Standard_ContentBasedFiltering
from ExerciseProject.RecommendSys.circuit.CollaborativeFiltering.P03Standard import Standard as Standard_CollaborativeFiltering

class Standard (
    Standard_CorrelationAnalysis
    , Standard_ContentBasedFiltering
    , Standard_CollaborativeFiltering
    ) :
    pass
