
class Dashboard() :

    @classmethod
    def AC0_0_1(self, functionInfo):
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt

        # 建立 Basemap 物件和其他繪圖設定
        m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='coral', lake_color='aqua')

        # 儲存圖形到圖像檔案
        plt.savefig("Example/P61MPlotDashboard/file/result/V0_0_1/None/AC0_0_1/test.png")

        return {}, {}

    @classmethod
    def AC0_0_2(self, functionInfo):
        from package.common.common.plotlib.MatplotlibDashboardCtrl import MatplotlibDashboardCtrl
        import numpy as np
        import pandas as pd

        basicData = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y1': [2, 4, 6, 8, 10],
            'y2': [1, 3, 5, 7, 9],
            'y3': [0, 2, 4, 6, 8],
            'y4': [0, 5, 6, 7, 8]
        })

        textInfo = {
            "textTitle": "測試文字"
            , "textLoc": "center"
            , "textSize": 24
        }

        tableInfo = {
            "tableData": basicData[['y1', 'y2', 'y3']].values
            , "columnLabelArr": ['y1', 'y2', 'y3']
            , "rowLabelArr": [1, 2, 3, 4, 5]
            , "labelHeight": 0.2
            , "dataHeight": 0.2
            , "colWidth": 0.2
        }

        radarInfo = {
            "radarTitle": ""
            , "radarData": basicData[['y1', 'y2', 'y3', 'y4']]
            , "radarMax": 10
            , "radarMin": -1
            , "isLegend": False
        }

        plotInfo = {
            "plotTitle": "Plot圖"
            , "plotXTitle": "AAAA"
            , "plotYTitle": "BBBB"
            , "plotLabels": ['y1', 'y2', 'y3', 'y4']
            , "plotTypes": basicData[['x']]
            , "plotValues": basicData[['y1', 'y2', 'y3', 'y4']]
            , "plotLine": {'y1': '-', 'y2': '-', 'y3': '-', 'y4': '-'}
        }

        pieInfo = {
            "ringTitle": "Pie圖"
            , "ringText": "中間文字"
            , "ringData": basicData['y1']
            , "ringColors": ['red', 'blue', 'green', 'yellow', 'black']
            , "ringLabels": None  # basicData['x']
        }

        barInfo = {
            "barTitle": "Bar圖"
            , "barXTitle": "X軸文字"
            , "barYTitle": "Y軸文字"
            , "barLabel": basicData['x'].values
            , "barValues": basicData['y1'].values
        }

        barhInfo = {
            "barhTitle": "Barh圖"
            , "barhXTitle": "X軸文字"
            , "barhYTitle": "Y軸文字"
            , "barhLabel": basicData['x'].values
            , "barhValues": basicData['y1'].values
        }

        matInfo = {
            "matTitle": ""
            , "columnLabelArr": ['S', 'M', 'T', 'W', 'T', 'F', 'S']
            ,"rowLabelArr": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17', '18', '19', '20', '21', '22', '23']
            , "matData": np.random.random((24, 7))
        }

        imageInfo = {
            "imagePath": "Example/P61MPlotDashboard/file/result/V0_0_1/None/AC0_0_1/test.png"
        }

        # 標準快速建立相關圖表
        figInfoArray = [
            (None, None, [[0, 1], [0, 1]])
            , (textInfo, "text", [[1, 2], [0, 1]])
            , (tableInfo, "table", [[0, 2], [1, 2]])
            , (radarInfo, "radar", [[2, 6], [0, 2]])
            , (plotInfo, "plot", [[0, 2], [2, 3]])
            , (pieInfo, "pie", [[0, 2], [3, 4]])
            , (barInfo, "bar", [[0, 2], [4, 5]])
            , (barhInfo, "barh", [[0, 2], [5, 6]])
            , (matInfo, "mat", [[0, 2], [6, 7]])
            , (imageInfo, "image", [[2, 6], [2, 6]])
        ]

        matplotlibDashboardCtrl = MatplotlibDashboardCtrl(figTitle="圖表範本", figsizeWidth=19, figsizeHigh=9,gridSpecHigh=6, gridSpecWidth=7)
        matplotlibDashboardCtrl.makeDashboard(figInfoArray)

        # 自由插入圖表
        fig , gs , cm = matplotlibDashboardCtrl.getDashboardFigGsCm()

        from matplotlib import ticker
        axInfo = [[2, 6], [6, 7]]
        plotAXInfo = matInfo
        plotAX = fig.add_subplot(gs[axInfo[0][0]:axInfo[0][1], axInfo[1][0]:axInfo[1][1]])
        matTitle = plotAXInfo["matTitle"] if "matTitle" in plotAXInfo.keys() else ""
        data = plotAXInfo["matData"] if "matData" in plotAXInfo.keys() else []
        cmap = plotAXInfo["cmap"] if "cmap" in plotAXInfo.keys() else 'Greys'
        plotAX.set(title=matTitle)
        pos = plotAX.matshow (data, interpolation='nearest', cmap=cmap)
        fig.colorbar(pos,ax=plotAX)
        plotAX.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plotAX.yaxis.set_major_locator(ticker.MultipleLocator(1))


        matplotlibDashboardCtrl.saveDashboard("Example/P61MPlotDashboard/file/result/V0_0_1/None/AC0_0_2/test.png")

        return {}, {}

