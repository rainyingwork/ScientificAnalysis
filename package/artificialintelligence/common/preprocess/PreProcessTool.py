
class PreProcessTool():

    @classmethod
    def filterXAllZeroColumn(self, fvInfo , preprossDF ):
        preprossDF["isAllZero"] = True
        for mdInfo in fvInfo['MakeDataInfo']:
            if mdInfo['DataType'] != "X":
                continue
            for columnNumber in mdInfo['ColumnNumbers']:
                columnFullName = str.lower("{}_{}_{}_{}_{}_{}".format(mdInfo["Product"], mdInfo["Project"], mdInfo['DTNameStr'], str(columnNumber), mdInfo['GFunc'], mdInfo["Version"]))
                preprossDF["isAllZero"] = preprossDF["isAllZero"] & (preprossDF[columnFullName] == 0)
        preprossDF = preprossDF[preprossDF["isAllZero"] == False]
        preprossDF = preprossDF.drop(columns=["isAllZero"])
        return fvInfo , preprossDF