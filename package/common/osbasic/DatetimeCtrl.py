import datetime
import calendar

class DatetimeCtrl ():

    def getISOCalendar (makeDatetime , firstweekday ) :
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

    def getThisWeekFirstDate (makeDatetime, firstweekday) :
        dateISOCalendar = getISOCalendar(makeDatetime, firstweekday)
        return makeDatetime - datetime.timedelta(days=dateISOCalendar[2]-1)

    def getThisMonthFirstDate (makeDatetime) :
        return datetime.datetime(makeDatetime.year, makeDatetime.month, 1, 0, 0, 0, 0)

    def getThisQuartFirstDate (makeDatetime) :
        if makeDatetime.month in (1, 2, 3):
            return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (4, 5, 6):
            return datetime.datetime(makeDatetime.year, 4, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (7, 8, 9):
            return datetime.datetime(makeDatetime.year, 7, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (10, 11, 12):
            return datetime.datetime(makeDatetime.year, 10, 1, 0, 0, 0, 0)

    def getThisHalfFirstDate (makeDatetime) :
        if makeDatetime.month in (1, 2, 3, 4, 5, 6):
            return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)
        elif makeDatetime.month in (7, 8, 9, 10, 11, 12):
            return datetime.datetime(makeDatetime.year, 7, 1, 0, 0, 0, 0)

    def getThisYearFirstDate (makeDatetime) :
        return datetime.datetime(makeDatetime.year, 1, 1, 0, 0, 0, 0)

    def getThisWeekLastDate (makeDatetime, firstweekday) :
        dateISOCalendar = getISOCalendar(makeDatetime, firstweekday)
        lastDateToWeek = makeDatetime + datetime.timedelta(days=(7 - dateISOCalendar[2]))
        return lastDateToWeek

    def getThisMonthLastDate (makeDatetime) :
        return datetime.datetime(makeDatetime.year, makeDatetime.month, calendar.monthrange(makeDatetime.year, makeDatetime.month)[1])

    def getThisQuartLastDate (makeDatetime) :
        if makeDatetime.month in (1, 2, 3):
            return datetime.datetime(makeDatetime.year, 3, calendar.monthrange(makeDatetime.year, 3)[1])
        elif makeDatetime.month in (4, 5, 6):
            return datetime.datetime(makeDatetime.year, 6, calendar.monthrange(makeDatetime.year, 6)[1])
        elif makeDatetime.month in (7, 8, 9):
            return datetime.datetime(makeDatetime.year, 9, calendar.monthrange(makeDatetime.year, 9)[1])
        elif makeDatetime.month in (10, 11, 12):
            return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getThisHalfLastDate (makeDatetime) :
        if makeDatetime.month in (1, 2, 3, 4, 5, 6):
            return datetime.datetime(makeDatetime.year, 6, calendar.monthrange(makeDatetime.year, 6)[1])
        elif makeDatetime.month in (7, 8, 9, 10, 11, 12):
            return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getThisYearLastDate (makeDatetime) :
        return datetime.datetime(makeDatetime.year, 12, calendar.monthrange(makeDatetime.year, 12)[1])

    def getPeriodThisFirstAndLastDate (makeDatetime ,periodtype, firstweekday) :
        if periodtype == "day" :
            return makeDatetime , makeDatetime
        if periodtype == "week" :
            return getThisWeekFirstDate (makeDatetime, firstweekday) , getThisWeekLastDate (makeDatetime, firstweekday)
        elif periodtype == "month" :
            return getThisMonthFirstDate (makeDatetime),  getThisMonthLastDate (makeDatetime)
        elif periodtype == "quart":
            return getThisQuartFirstDate (makeDatetime), getThisQuartLastDate (makeDatetime)
        elif periodtype == "half":
            return getThisHalfFirstDate (makeDatetime), getThisHalfLastDate (makeDatetime)
        elif periodtype == "year":
            return  getThisYearFirstDate (makeDatetime), getThisYearLastDate (makeDatetime)

    def getRangeAllDateArr (startDateTime , endDateTime) :
        makeTimeArr = []
        makeDatetime = startDateTime
        while makeDatetime <= endDateTime:
            makeTimeArr.append(makeDatetime)
            makeDatetime = makeDatetime + datetime.timedelta(days=1)
        return makeTimeArr

    def getRangeDateInfoArr (startDateTime , endDateTime , periodtype , firstweekday=None) :
        dateInfoArr = []
        if periodtype == "day":
            makeDatetime = startDateTime
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": makeDatetime.strftime("%Y-%m-%d")
                    , "startDateTime": makeDatetime
                    , "endDateTime": makeDatetime
                    , "allDateArr": getRangeAllDateArr(makeDatetime , endDateTime)
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = makeDatetime + datetime.timedelta(days=1)
        elif periodtype == "week":
            makeDatetime = getThisWeekFirstDate(startDateTime, firstweekday)
            while makeDatetime <= endDateTime:
                dateISOCalendar = getISOCalendar(makeDatetime, firstweekday)
                dateInfo = {
                    "dateName": "W{}{}".format(dateISOCalendar[0],dateISOCalendar[1])
                    , "startDateTime": makeDatetime
                    , "endDateTime": makeDatetime + datetime.timedelta(days=6)
                    , "allDateArr": getRangeAllDateArr(makeDatetime , makeDatetime + datetime.timedelta(days=6))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = makeDatetime + datetime.timedelta(days=7)
        elif periodtype == "month":
            makeDatetime = getThisMonthFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "M{}".format(makeDatetime.strftime("%Y%m"))
                    , "startDateTime": makeDatetime
                    , "endDateTime": getThisMonthLastDate(makeDatetime)
                    , "allDateArr": getRangeAllDateArr(makeDatetime, getThisMonthLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = getThisMonthLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "quart":
            makeDatetime = getThisQuartFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "Q{}{}".format(makeDatetime.strftime("%Y"),(makeDatetime.month //3 + 1 ).zfill(2))
                    , "startDateTime": makeDatetime
                    , "endDateTime": getThisQuartLastDate(makeDatetime)
                    , "allDateArr": getRangeAllDateArr(makeDatetime, getThisQuartLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = getThisQuartLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "half":
            makeDatetime = getThisHalfFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "H{}{}".format(makeDatetime.strftime("%Y"),(makeDatetime.month //6 + 1).zfill(2))
                    , "startDateTime": makeDatetime
                    , "endDateTime": getThisHalfLastDate(makeDatetime)
                    , "allDateArr": getRangeAllDateArr(makeDatetime, getThisHalfLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = getThisHalfLastDate(makeDatetime) + datetime.timedelta(days=1)
        elif periodtype == "year":
            makeDatetime = getThisYearFirstDate(startDateTime)
            while makeDatetime <= endDateTime:
                dateInfo = {
                    "dateName": "Y{}".format(makeDatetime.strftime("%Y"))
                    , "startDateTime": makeDatetime
                    , "endDateTime": getThisYearLastDate(makeDatetime)
                    , "allDateArr": getRangeAllDateArr(makeDatetime, getThisYearLastDate(makeDatetime))
                }
                dateInfoArr.append(dateInfo)
                makeDatetime = getThisYearLastDate(makeDatetime) + datetime.timedelta(days=1)

        return dateInfoArr