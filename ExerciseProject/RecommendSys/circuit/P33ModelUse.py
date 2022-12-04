from ExerciseProject.RecommendSys.circuit.CorrelationAnalysis.P33ModelUse import ModelUse as ModelUse_CorrelationAnalysis
from ExerciseProject.RecommendSys.circuit.ContentBasedFiltering.P33ModelUse import ModelUse as ModelUse_ContentBasedFiltering
from ExerciseProject.RecommendSys.circuit.CollaborativeFiltering.P33ModelUse import ModelUse as ModelUse_CollaborativeFiltering

class ModelUse(
    ModelUse_CorrelationAnalysis
    , ModelUse_ContentBasedFiltering
    , ModelUse_CollaborativeFiltering
    ) :
    pass
