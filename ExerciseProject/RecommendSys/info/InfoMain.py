from ExerciseProject.RecommendSys.info.P01CrawlerInfo import CrawlerInfo
from ExerciseProject.RecommendSys.info.P02OriginalInfo import OriginalInfo
from ExerciseProject.RecommendSys.info.P03StandardInfo import StandardInfo
from ExerciseProject.RecommendSys.info.P31RawDataInfo import RawDataInfo
from ExerciseProject.RecommendSys.info.P32PreProcessInfo import PreProcessInfo
from ExerciseProject.RecommendSys.info.P33ModelUseInfo import ModelUseInfo

class InfoMain(
    CrawlerInfo
    , OriginalInfo
    , StandardInfo
    , RawDataInfo
    , PreProcessInfo
    , ModelUseInfo
    ) :
    pass

