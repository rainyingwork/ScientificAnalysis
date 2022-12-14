from Example.P23Standard.info.P01CrawlerInfo import CrawlerInfo
from Example.P23Standard.info.P02OriginalInfo import OriginalInfo
from Example.P23Standard.info.P03StandardInfo import StandardInfo
from Example.P23Standard.info.P31RawDataInfo import RawDataInfo
from Example.P23Standard.info.P32PreProcessInfo import PreProcessInfo
from Example.P23Standard.info.P33ModelUseInfo import ModelUseInfo

class InfoMain(
    CrawlerInfo
    , OriginalInfo
    , StandardInfo
    , RawDataInfo
    , PreProcessInfo
    , ModelUseInfo
    ) :
    pass

