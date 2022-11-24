import pandas
import time
import asyncio

class AsyncioCtrl:

    def __init__(self):
        self._loop = asyncio.get_event_loop()
        self._startTime = time.time()

    def runAsyncio(self, asyncioInfoArr):
        if asyncioInfoArr != []:
            tasks = []
            semaphore = asyncio.Semaphore(20)
            count = 0
            for asyncioInfo in asyncioInfoArr:
                count = count + 1
                asyncioInfo["semaphore"] = semaphore
                asyncioInfo["asynciocount"] = count
                task = asyncio.ensure_future(self.send_req(asyncioInfo))
                tasks.append(task)
            self._loop.run_until_complete(asyncio.wait(tasks))

    async def send_req(self,asyncioInfo ):
        semaphore = asyncioInfo["semaphore"] if "semaphore" in asyncioInfo.keys() else []
        asynciocount = asyncioInfo["asynciocount"] if "asynciocount" in asyncioInfo.keys() else []
        func = asyncioInfo["func"] if "func" in asyncioInfo.keys() else []
        funcargs = asyncioInfo["funcargs"] if "funcargs" in asyncioInfo.keys() else []
        async with semaphore:
            endTime = time.time()
            print("Send a request at", endTime - self._startTime, "seconds. {}".format(str(asynciocount)))
            try:
                res = await self._loop.run_in_executor(None,func,funcargs)
            except:
                print("error {}".format(str(asynciocount)))
            endTime = time.time()
            print("Receive a response at", endTime - self._startTime, "seconds. {}".format(str(asynciocount)))
