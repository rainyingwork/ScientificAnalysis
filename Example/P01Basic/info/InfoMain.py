from Example.P01Basic.info.P01CrawlerInfo import CrawlerInfo
from Example.P01Basic.info.P02OriginalInfo import OriginalInfo
from Example.P01Basic.info.P03StandardInfo import StandardInfo
from Example.P01Basic.info.P31RawDataInfo import RawDataInfo
from Example.P01Basic.info.P32PreProcessInfo import PreProcessInfo
from Example.P01Basic.info.P33ModelUseInfo import ModelUseInfo

class InfoMain(
    CrawlerInfo
    , OriginalInfo
    , StandardInfo
    , RawDataInfo
    , PreProcessInfo
    , ModelUseInfo
    ) :
    pass

