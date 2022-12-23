import datetime
import calendar

class DatetimeCtrl ():

    def getISOCalendar (self , makeDatetime , firstweekday ) :
        firstDayBasicToYear = datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0).weekday() - (firstweekday - 1 )
        firstDayBasicToYear = firstDayBasicToYear + 7 if firstDayBasicToYear < 0 else firstDayBasicToYear
        countDayToYear = (makeDatetime - datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)).days
        day = (countDayToYear + firstDayBasicToYear) % 7 + 1
        if makeDatetime.year != (makeDatetime + datetime.timedelta(days=(7-day))).year :
            year = makeDatetime.year + 1
            week = 1
        else :
            year = makeDatetime.year
            week = (countDayToYear + firstDayBasicToYear) // 7 + 1
        return (year,week,day)

    def getThisWeekFirstDate (self ,makeDatetime, firstweekday) :
        dateISOCalendar = self.getISOCalendar(makeDatetime, firstweekday)
        return makeDatetime - datetime.timedelta(days=dateISOCalendar[2]-1)

    def getThisMonthFirstDate (self ,makeDatetime) :
        return datetime.datetime(makeDatetime.year, makeDatetime.month, 1, 0, 0, 0, 0)

    def getThisQuartFirstDate (self ,makeDatetime) :
        if makeDatetime.month in (1, 2, 3):
            return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (4, 5, 6):
            return datetime.datetime(makeDatetime.year, 4, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (7, 8, 9):
            return datetime.datetime(makeDatetime.year, 7, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (10, 11, 12):
            return datetime.datetime(makeDatetime.year, 10, 1, 0, 0, 0, 0)

    def getThisHalfFirstDate (self ,makeDatetime) :
        if makeDatetime.month in (1, 2, 3, 4, 5, 6):
            return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (7, 8, 9, 10, 11, 12):
            return datetime.datetime(makeDatetime.year, 7, 1, 0, 0, 0, 0)

    def getThisYearFirstDate (self ,makeDatetime) :
        return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)

    def getThisWeekLastDate (self ,makeDatetime, firstweekday) :
        dateISOCalendar = self.getISOCalendar(makeDatetime, firstweekday)
        lastDateToWeek = makeDatetime + datetime.timedelta(days=(7 - dateISOCalendar[2]))
        return lastDateToWeek

    def getThisMonthLastDate (self ,makeDatetime) :
        return datetime.datetime(makeDatetime.year, makeDatetime.month, calendar.monthrange(makeDatetime.year, makeDatetime.month)[1])

    def getThisQuartLastDate (self ,makeDatetime) :
        if makeDatetime.month in (1, 2, 3):
            return datetime.datetime(makeDatetime.year, 3, calendar.monthrange(makeDatetime.year, 3)[1])
        elif makeDatetime.month in (4, 5, 6):
            return datetime.datetime(makeDatetime.year, 6, calendar.monthrange(makeDatetime.year, 6)[1])
        elif makeDatetime.month in (7, 8, 9):
            return datetime.datetime(makeDatetime.year, 9, calendar.monthrange(makeDatetime.year, 9)[1])
        elif makeDatetime.month in (10, 11, 12):
            return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getThisHalfLastDate (self ,makeDatetime) :
        if makeDatetime.month in (1, 2, 3, 4, 5, 6):
            return datetime.datetime(makeDatetime.year, 6, calendar.monthrange(makeDatetime.year, 6)[1])
        elif makeDatetime.month in (7, 8, 9, 10, 11, 12):
            return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getThisYearLastDate (self ,makeDatetime) :
        return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getPeriodThisFirstAndLastDate (self ,makeDatetime ,periodtype, firstweekday) :
        if periodtype == "day" :
            return makeDatetime , makeDatetime
        if periodtype == "week" :
            return self.getThisWeekFirstDate (makeDatetime, firstweekday) , self.getThisWeekLastDate (makeDatetime, firstweekday)
        elif periodtype == "month" :
            return self.getThisMonthFirstDate (makeDatetime),  self.getThisMonthLastDate (makeDatetime)
        elif periodtype == "quart":
            return self.getThisQuartFirstDate (makeDatetime), self.getThisQuartLastDate (makeDatetime)
        elif periodtype == "half":
            return self.getThisHalfFirstDate (makeDatetime), self.getThisHalfLastDate (makeDatetime)
        elif periodtype == "year":
            return  self.getThisYearFirstDate (makeDatetime), self.getThisYearLastDate (makeDatetime)

    def getRangeAllDateArr (self ,startDateTime , endDateTime) :
        makeTimeArr = []
        makeDatetime = startDateTime
        while makeDatetime <= endDateTime:
            makeTimeArr.append(makeDatetime)
            makeDatetime = makeDatetime + datetime.timedelta(days=1)
        return makeTimeArr

    def getRangeDateInfoArr (self ,startDateTime , endDateTime , periodtype , firstweekday=None) :
        dateInfoArr = []
        if periodtype == "day":
            makeDatetime = startDateTime
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": makeDatetime.strftime("%Y-%m-%d")
                    , "startDateTime": makeDatetime
                    , "endDateTime": makeDatetime
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime , endDateTime)
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = makeDatetime + datetime.timedelta(days=1)
        elif periodtype == "week":
            makeDatetime = self.getThisWeekFirstDate(startDateTime, firstweekday)
            while makeDatetime <= endDateTime:
                dateISOCalendar = self.getISOCalendar(makeDatetime, firstweekday)
                dateInfo = {
                    "dateName": "W{}{}".format(dateISOCalendar[0],dateISOCalendar[1])
                    , "startDateTime": makeDatetime
                    , "endDateTime": makeDatetime + datetime.timedelta(days=6)
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime , makeDatetime + datetime.timedelta(days=6))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = makeDatetime + datetime.timedelta(days=7)
        elif periodtype == "month":
            makeDatetime = self.getThisMonthFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "M{}".format(makeDatetime.strftime("%Y%m"))
                    , "startDateTime": makeDatetime
                    , "endDateTime": self.getThisMonthLastDate(makeDatetime)
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime, self.getThisMonthLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = self.getThisMonthLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "quart":
            makeDatetime = self.getThisQuartFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "Q{}{}".format(makeDatetime.strftime("%Y"),(makeDatetime.month //3 + 1 ).zfill(2))
                    , "startDateTime": makeDatetime
                    , "endDateTime": self.getThisQuartLastDate(makeDatetime)
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime, self.getThisQuartLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = self.getThisQuartLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "half":
            makeDatetime = self.getThisHalfFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "H{}{}".format(makeDatetime.strftime("%Y"),(makeDatetime.month //6 + 1).zfill(2))
                    , "startDateTime": makeDatetime
                    , "endDateTime": self.getThisHalfLastDate(makeDatetime)
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime, self.getThisHalfLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = self.getThisHalfLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "year":
            makeDatetime = self.getThisYearFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "Y{}".format(makeDatetime.strftime("%Y"))
                    , "startDateTime": makeDatetime
                    , "endDateTime": self.getThisYearLastDate(makeDatetime)
                    , "allDateArr": self.getRangeAllDateArr(makeDatetime, self.getThisYearLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = self.getThisYearLastDate(makeDatetime) + datetime.timedelta(days=1)

        return dateInfoArr