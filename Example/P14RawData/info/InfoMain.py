from Example.P14RawData.info.P01CrawlerInfo import CrawlerInfo
from Example.P14RawData.info.P02OriginalInfo import OriginalInfo
from Example.P14RawData.info.P03StandardInfo import StandardInfo
from Example.P14RawData.info.P31RawDataInfo import RawDataInfo
from Example.P14RawData.info.P32PreProcessInfo import PreProcessInfo
from Example.P14RawData.info.P33ModelUseInfo import ModelUseInfo

class InfoMain(
    CrawlerInfo
    , OriginalInfo
    , StandardInfo
    , RawDataInfo
    , PreProcessInfo
    , ModelUseInfo
    ) :
    pass

