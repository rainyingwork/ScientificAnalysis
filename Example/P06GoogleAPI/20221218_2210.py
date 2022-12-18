import time
from package.googleapi.common.slide.GoogleSlideAPICtrl import GoogleSlideAPICtrl

googleSlideAPICtrl = GoogleSlideAPICtrl(
    presentationId = '1TzoUhDUZAxpL82BmBsx9HsKcmMZA6mkJ48ny_WM14V8'
    , scopes = ['https://www.googleapis.com/auth/drive'
          , 'https://www.googleapis.com/auth/presentations'
          , 'https://www.googleapis.com/auth/spreadsheets']
    , credentialPath = 'env/credentials.json'
    , tokenPath = 'env/token.json'
)

enumNameArr = [
    'MAIN_POINT' # 要點
    , 'TITLE' # 主標題
    , 'SECTION_HEADER' # 章節標題
    , 'TITLE_ONLY' # 標題
    , 'TITLE_AND_BODY' # 標題與內文
    , 'ONE_COLUMN_TEXT'  # 單欄文字
    , 'TITLE_AND_TWO_COLUMNS' # 標題與兩欄
    , 'BIG_NUMBER'  # 大數字
    , 'CAPTION_ONLY' # 圖說
    , 'SECTION_TITLE_AND_DESCRIPTION'  # 章節標題與說明
    , 'BLANK' # 空白
]

# enumNameArr = [
#     'TITLE' # 主標題
#     , 'BIG_NUMBER' # 大數字
#     , 'MAIN_POINT' # 要點
#     , 'CAPTION_ONLY' # 圖說
#     , 'TITLE_AND_BODY'  # 標題與內文
#     , 'TITLE_AND_TWO_COLUMNS' # 標題與兩欄
#     , 'TITLE_ONLY'  # 標題
#     , 'SECTION_HEADER' # 章節標題
#     , 'SECTION_TITLE_AND_DESCRIPTION' # 章節標題與說明
#     , 'ONE_COLUMN_TEXT' # 單欄文字
# ]


# for pageIndex in range(0,len(enumNameArr)):
#     pageObjectId = str(round(time.time() * 1000))
#     slideRequests =googleSlideAPICtrl.getRequestsArrByCreatePage(pageObjectId,pageIndex,enumNameArr[pageIndex])
#     googleSlideAPICtrl.exeSlideRequests(slideRequests)

slideInfo = {
    'PageObjectId': str(round(time.time() * 1000))
    , 'PageIndex': 0
    , 'TitleObjectId': str(round(time.time() * 1000)+1)
    , 'TitleInsertText': 'TitleInsertText'
    , 'BodyObjectId': str(round(time.time() * 1000)+2)
    , 'BodyInsertText': 'BodyInsertText'
}
slideRequests = googleSlideAPICtrl.getRequestsArrByCreateTitleAndBody(slideInfo)
googleSlideAPICtrl.exeSlideRequests(slideRequests)

