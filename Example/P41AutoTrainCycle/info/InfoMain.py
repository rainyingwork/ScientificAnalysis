from Example.P41AutoTrainCycle.info.P01CrawlerInfo import CrawlerInfo
from Example.P41AutoTrainCycle.info.P02OriginalInfo import OriginalInfo
from Example.P41AutoTrainCycle.info.P03StandardInfo import StandardInfo
from Example.P41AutoTrainCycle.info.P31RawDataInfo import RawDataInfo
from Example.P41AutoTrainCycle.info.P32PreProcessInfo import PreProcessInfo
from Example.P41AutoTrainCycle.info.P33ModelUseInfo import ModelUseInfo

class InfoMain(
    CrawlerInfo
    , OriginalInfo
    , StandardInfo
    , RawDataInfo
    , PreProcessInfo
    , ModelUseInfo
    ) :
    pass

