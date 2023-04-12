import os
import time
import json
import Config
from datetime import datetime
from dotenv import load_dotenv
from package.common.common.database.PostgresCtrl import PostgresCtrl
from package.common.common.message.TelegramCtrl import TelegramCtrl
load_dotenv(dotenv_path="env/postgresql.env")
telegramCtrl = TelegramCtrl(env="env/telegram.env")
postgresCtrl = PostgresCtrl(
    host=os.getenv("POSTGRES_HOST")
    , port=int(os.getenv("POSTGRES_POST"))
    , user=os.getenv("POSTGRES_USERNAME")
    , password=os.getenv("POSTGRES_PASSWORD")
    , database=os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"]
    , schema=os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"]
)

while True:
    try:
        isRunOPS = False

        # ================================================== 檢查有沒有新的訊息 ==================================================

        messageJsonStr = telegramCtrl.findMessage()
        messageJson = json.loads(messageJsonStr)
        updateIdSQL = """
            select 
                MAX(resultjson::json ->> 'UpdateID') as UpdateID 
            from opsmanagement.opsdetail  
            where 1 = 1 
                and exefunction = 'Chat0_0_1'
                and state = 'FINISH'
        """
        updateIdDF = postgresCtrl.searchSQL(updateIdSQL)
        updateId = int(updateIdDF['updateid'][0])
        for messageDict in messageJson['result']:
            if messageDict['update_id'] > updateId:
                isRunOPS = True

        # ================================================== 檢查有沒有待處理的訊息 ==================================================

        updateIdSQL = """
            with Chat0_0_1 as ( 
                select 
                    resultjson::json ->> 'UpdateID' as UpdateID
                    , AA.*
                from opsmanagement.opsdetail AA  
                where 1 = 1 
                    and exefunction = 'Chat0_0_1'
                    and state = 'FINISH'
            ) , Chat0_0_2 as ( 
                select 
                    resultjson::json ->> 'UpdateID' as UpdateID
                from opsmanagement.opsdetail  
                where 1 = 1 
                    and exefunction = 'Chat0_0_2'
                    and state = 'FINISH'
            ) select 
                AA.resultjson
            from Chat0_0_1 AA  
            LEFT join Chat0_0_2 BB on 1 = 1
                and AA.UpdateID = BB.UpdateID
            where 1 = 1 
                and AA.UpdateID is not null
                and BB.UpdateID is null
            order by 
                AA.UpdateID ASC
        """

        updateDF = postgresCtrl.searchSQL(updateIdSQL)

        if len(updateDF) > 0:
            isRunOPS = True

        # ================================================== 執行OPS ==================================================

        if isRunOPS == True:
            print("isRunOPS is True {}".format(datetime.now()))
            import copy
            import OPSCommon as executeOPSCommon

            basicInfo = {
                "RunType": ["RunOPS"],
                "Product": ["Example"],
                "Project": ["P72Telegram"],
            }
            opsInfo = copy.deepcopy(basicInfo)
            opsInfo["OPSVersion"] = ["V0_0_1"]
            opsInfo["OPSOrderJson"] = {
                "ExeFunctionArr": ["Chat0_0_1", "Chat0_0_2"],
                # "RepOPSRecordId": 0,
                # "RepFunctionArr": [""],
                # "RunFunctionArr": [""],
                "OrdFunctionArr": [
                    {"Parent": "Chat0_0_1", "Child": "Chat0_0_2"},
                ],
                "FunctionMemo": {
                    "Chat0_0_1": "訊息保存",
                    "Chat0_0_2": "訊息回覆",
                },
            }
            opsInfo["ParameterJson"] = {}
            opsInfo["ResultJson"] = {}

            executeOPSCommon.main(opsInfo)
        else :
            print("isRunOPS is False {}".format(datetime.now()))
        time.sleep(10)
    except :
        print("Error {}".format(datetime.now()))
        time.sleep(10)