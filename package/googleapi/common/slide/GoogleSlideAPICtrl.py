import os
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleSlideAPICtrl ():

    def __init__(self , presentationId , scopes ,credentialPath, tokenPath ):
        self.presentationId = presentationId
        self.scopes = scopes
        self.creds = None
        if os.path.exists(tokenPath):
            self.creds = Credentials.from_authorized_user_file(tokenPath, self.scopes)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentialPath, self.scopes)
                self.creds = flow.run_local_server(port=0)
            with open(tokenPath, 'w') as token:
                token.write(self.creds.to_json())


    def exeSlideRequests(self,slideRequests) :
        try:
            service = build('slides', 'v1', credentials=self.creds)
            body = {'requests': slideRequests}
            response = service.presentations().batchUpdate(presentationId=self.presentationId, body=body).execute()
            response.get('replies')[0].get('createSlide')
        except HttpError as err:
            print(err)

    @classmethod
    def getRequestsArrByCreatePage(self, pageObjectId ,pageIndex, layoutEnum):
        slideRequests = [
            {
                'createSlide': {
                    'objectId': pageObjectId,
                    'insertionIndex': pageIndex ,
                    'slideLayoutReference': {
                        'predefinedLayout': layoutEnum
                    }
                }
            }
        ]
        return slideRequests

    @classmethod
    def getRequestsArrByCreateTitleAndBody(self, slideInfo):
        pageObjectId = slideInfo.get('PageObjectId') if slideInfo.get('PageObjectId') else str(round(time.time() * 1000))
        pageIndex = slideInfo.get('PageIndex') if slideInfo.get('PageIndex') else 0
        titleObjectId = slideInfo.get('TitleObjectId') if slideInfo.get('TitleObjectId') else str(round(time.time() * 1000))
        titleInsertText = slideInfo.get('TitleInsertText') if slideInfo.get('TitleInsertText') else "TitleInsertText"
        bodyObjectId = slideInfo.get('BodyObjectId') if slideInfo.get('BodyObjectId') else str(round(time.time() * 1000))
        bodyInsertText = slideInfo.get('BodyInsertText') if slideInfo.get('BodyInsertText') else "BodyInsertText"

        slideRequests = [
            {
                'createSlide': {
                    'objectId': pageObjectId
                    , 'insertionIndex': pageIndex
                    , 'slideLayoutReference': {
                        'predefinedLayout': 'BLANK'
                    }
                }
            }
            , {
                'createShape': {
                    'objectId': titleObjectId
                    , 'shapeType': 'TEXT_BOX'
                    , 'elementProperties': {
                        'pageObjectId': pageObjectId
                        , 'size': {
                            'width': {'magnitude': 21.4 * (100 / 3.53), 'unit': 'PT'}
                            , 'height': {'magnitude': 2.5 * (100 / 3.53), 'unit': 'PT'}
                        }
                        , 'transform': {
                            'scaleX': 1
                            , 'scaleY': 1
                            , 'translateX': 2 * (100 / 3.53)
                            , 'translateY': 1 * (100 / 3.53)
                            , 'unit': 'PT'
                        }
                    }
                }
            }
            , {
                'insertText': {
                    'objectId': titleObjectId
                    , 'insertionIndex': 0
                    , 'text': titleInsertText
                }
            }
            , {
                'updateTextStyle': {
                    'objectId': titleObjectId
                    , 'style': {
                        'fontFamily': 'Share Tech'
                        , 'fontSize': {
                            'magnitude': 40
                            , 'unit': 'PT'
                        }
                        , 'foregroundColor': {
                            'opaqueColor': {
                                'rgbColor': {
                                    'blue': 1.0
                                    , 'green': 1.0
                                    , 'red': 1.0
                                }
                            }
                        }
                    }
                    , 'fields': 'foregroundColor,fontFamily,fontSize'
                }
            }
            , {
                'createShape': {
                    'objectId': bodyObjectId
                    , 'shapeType': 'TEXT_BOX'
                    , 'elementProperties': {
                        'pageObjectId': pageObjectId
                        , 'size': {
                            'width': {'magnitude': 21.4 * (100 / 3.53), 'unit': 'PT'}
                            , 'height': {'magnitude': 9 * (100 / 3.53), 'unit': 'PT'}
                        }
                        , 'transform': {
                            'scaleX': 1
                            , 'scaleY': 1
                            , 'translateX': 2 * (100 / 3.53)
                            , 'translateY': 4 * (100 / 3.53)
                            , 'unit': 'PT'
                        }

                    }
                }
            }
            , {
                'insertText': {
                    'objectId': bodyObjectId
                    , 'insertionIndex': 0
                    , 'text': bodyInsertText
                }
            }
            , {
                'updateTextStyle': {
                    'objectId': bodyObjectId
                    , 'style': {
                        'fontFamily': 'Share Tech'
                        , 'fontSize': {
                            'magnitude': 14
                            , 'unit': 'PT'
                        }
                        , 'foregroundColor': {
                            'opaqueColor': {
                                'rgbColor': {
                                    'blue': 1.0
                                    , 'green': 1.0
                                    , 'red': 1.0
                                }
                            }
                        }
                    }
                    , 'fields': 'foregroundColor,fontFamily,fontSize'
                }
            }
            , {
                'createSheetsChart': {
                    'objectId': str(round(time.time() * 1000)),
                    'spreadsheetId': str(round(time.time() * 1000)+1),
                    'chartId': str(round(time.time() * 1000)+2),
                    'linkingMode': 'LINKED',
                    'elementProperties': {
                        'pageObjectId': pageObjectId,
                        'size': {
                            'height': {'magnitude': 4000000,'unit': 'EMU'},
                            'width': {'magnitude': 4000000,'unit': 'EMU'}
                        },
                        'transform': {
                            'scaleX': 1,
                            'scaleY': 1,
                            'translateX': 100000,
                            'translateY': 100000,
                            'unit': 'EMU'
                        }
                    }
                }
            }
        ]
        return slideRequests







