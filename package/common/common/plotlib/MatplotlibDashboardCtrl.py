from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

class MatplotlibDashboardCtrl:

    def __init__(self, figTitle, figsizeWidth, figsizeHigh, gridSpecHigh, gridSpecWidth):
        plt.rcParams['font.sans-serif'] = ['DFKai-SB']
        plt.rcParams['axes.unicode_minus'] = False
        self.figsizeWidth = figsizeWidth
        self.figsizeHigh = figsizeHigh
        self.gridSpecHigh = gridSpecHigh
        self.gridSpecWidth = gridSpecWidth
        self.fig = plt.figure(figsize=(figsizeWidth, figsizeHigh))
        self.gs = GridSpec(gridSpecHigh, gridSpecWidth)
        self.fig.suptitle(figTitle,fontsize= 24, color='k',fontweight='bold')
        self.cm = plt.get_cmap("tab20c")

    def makeDashboard(self, figInfoArray):
        for data, type, axInfo in figInfoArray :
            ax = self.fig.add_subplot(self.gs[axInfo[0][0]:axInfo[0][1], axInfo[1][0]:axInfo[1][1]])
            if type == "plot":
                self.makeDashboardPlot(ax, data)
            elif type == "pie":
                self.makeDashboardPie(ax, data)
            elif type == "bar":
                self.makeDashboardBar(ax, data)
            elif type == "barh":
                self.makeDashboardBarh(ax, data)
            elif type == "table":
                self.makeDashboardTable(ax, data)
            elif type == "mat":
                self.makeDashboardMat(ax, data)

    def slowDashboard(self):
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+50+50")
        plt.tight_layout()
        plt.show()

    def makeDashboardPlot(self,plotAX, plotAXInfo):
        plotTitle = plotAXInfo["plotTitle"] if "plotTitle" in plotAXInfo.keys() else ""
        plotXTitle = plotAXInfo["plotXTitle"] if "plotXTitle" in plotAXInfo.keys() else ""
        plotYTitle = plotAXInfo["plotYTitle"] if "plotYTitle" in plotAXInfo.keys() else ""
        plotLabels = plotAXInfo["plotLabels"] if "plotLabels" in plotAXInfo.keys() else []
        plotTypes = plotAXInfo["plotTypes"] if "plotTypes" in plotAXInfo.keys() else []
        plotValues = plotAXInfo["plotValues"] if "plotValues" in plotAXInfo.keys() else []
        plotLine = plotAXInfo["plotLine"] if "plotLine" in plotAXInfo.keys() else []
        for labelName in plotLabels:
            plotAX.plot(plotTypes, plotValues[labelName], plotLine[labelName], label=labelName)
        plotAX.legend(plotLabels)
        plotAX.set(title=plotTitle ,xlabel=plotXTitle,ylabel=plotYTitle)
        plotAX.spines['right'].set_visible(False)
        plotAX.spines['top'].set_visible(False)

    def makeDashboardPie(self,plotAX, plotAXInfo):
        ringRadius, ringWidth = 2, 0.8
        ringTitle = plotAXInfo["ringTitle"] if "ringTitle" in plotAXInfo.keys() else ""
        ringText = plotAXInfo["ringText"] if "ringText" in plotAXInfo.keys() else ""
        ringData = plotAXInfo["ringData"] if "ringData" in plotAXInfo.keys() else []
        ringColors = plotAXInfo["ringColors"] if "ringText" in plotAXInfo.keys() else np.arange(len(ringData))
        ringLabels = plotAXInfo["ringLabels"] if "ringLabels" in plotAXInfo.keys() else self.cm(np.array(np.arange(len(ringData))))

        pieOut, _ = plotAX.pie(ringData, radius=ringRadius, colors=ringColors, startangle=90, rotatelabels=True)
        plt.setp(pieOut, width=ringWidth, edgecolor='white')
        plotAX.set(title=ringTitle)
        plotAX.legend(ringLabels, loc=(0.16 + 0.01 * self.figsizeWidth, -0.20), fontsize=8)
        plotAX.text(0., 0., ringText, horizontalalignment='center', verticalalignment='center')
        plotAX.axis('equal')

    def makeDashboardBar(self,plotAX, plotAXInfo):
        barTitle = plotAXInfo["barTitle"] if "barTitle" in plotAXInfo.keys() else ""
        barXTitle = plotAXInfo["barXTitle"] if "barXTitle" in plotAXInfo.keys() else ""
        barYTitle = plotAXInfo["barYTitle"] if "barYTitle" in plotAXInfo.keys() else ""
        barLabel = plotAXInfo["barLabel"] if "barLabel" in plotAXInfo.keys() else []
        barValues = plotAXInfo["barValues"] if "barValues" in plotAXInfo.keys() else []
        plotAX.set(title=barTitle,xlabel=barXTitle,ylabel=barYTitle)
        plotAX.bar(barLabel, barValues, color=self.cm(np.array([0])), linewidth=0)
        plotAX.spines['right'].set_visible(False)
        plotAX.spines['top'].set_visible(False)

    def makeDashboardBarh(self,plotAX, plotAXInfo):
        barhTitle = plotAXInfo["barhTitle"] if "barhTitle" in plotAXInfo.keys() else ""
        barhXTitle = plotAXInfo["barhXTitle"] if "barhXTitle" in plotAXInfo.keys() else ""
        barhYTitle = plotAXInfo["barhYTitle"] if "barhYTitle" in plotAXInfo.keys() else ""
        barhLabel = plotAXInfo["barhLabel"] if "barhLabel" in plotAXInfo.keys() else []
        barhValues = plotAXInfo["barhValues"] if "barhValues" in plotAXInfo.keys() else []
        plotAX.set(title=barhTitle,xlabel=barhXTitle,ylabel=barhYTitle)
        plotAX.barh(barhLabel, barhValues, color=self.cm(np.array([17])), linewidth=0)
        plotAX.spines['right'].set_visible(False)
        plotAX.spines['top'].set_visible(False)

    def makeDashboardTable(self,plotAX, plotAXInfo):
        tableData = plotAXInfo["tableData"] if "tableData" in plotAXInfo.keys() else []
        columnLabelArr = plotAXInfo["columnLabelArr"] if "columnLabelArr" in plotAXInfo.keys() else []
        rowLabelArr = plotAXInfo["rowLabelArr"] if "rowLabelArr" in plotAXInfo.keys() else []
        labelHeight = plotAXInfo["labelHeight"] if "labelHeight" in plotAXInfo.keys() else 0.3
        dataHeight = plotAXInfo["dataHeight"] if "dataHeight" in plotAXInfo.keys() else 0.2
        colWidth = plotAXInfo["colWidth"] if "colWidth" in plotAXInfo.keys() else 0.2

        df = pd.DataFrame(tableData, columns=columnLabelArr)
        plotAX.axis('off')
        plotTable = plotAX.table(
            cellText=df.values
            , cellLoc='center'
            , colLabels=columnLabelArr
            , colLoc='center'
            , colWidths=[colWidth for x in columnLabelArr]
            , rowLabels=rowLabelArr
            , rowLoc='center'
            , loc='center'
        )
        cellDict = plotTable.get_celld()
        for i in range(0, len(columnLabelArr)):
            cellDict[(0, i)].set_height(labelHeight)
            for j in range(1, len(tableData)+1):
                cellDict[(j, i)].set_height(dataHeight)
        for j in range(1, len(tableData)+1):
            cellDict[(j, -1)].set_height(dataHeight)

    def makeDashboardMat(self, plotAX, plotAXInfo):
        matTitle = plotAXInfo["matTitle"] if "matTitle" in plotAXInfo.keys() else ""
        data = plotAXInfo["matData"] if "matData" in plotAXInfo.keys() else []
        columnLabelArr = plotAXInfo["columnLabelArr"] if "columnLabelArr" in plotAXInfo.keys() else []
        rowLabelArr = plotAXInfo["rowLabelArr"] if "rowLabelArr" in plotAXInfo.keys() else []
        cmap = plotAXInfo["cmap"] if "cmap" in plotAXInfo.keys() else 'Greys'
        plotAX.set(title=matTitle)
        pos = plotAX.matshow (data, interpolation='nearest', cmap=cmap)
        self.fig.colorbar(pos,ax=plotAX)
        plotAX.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plotAX.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plotAX.set_xticklabels([''] + columnLabelArr, fontsize=8)
        plotAX.set_yticklabels([''] + rowLabelArr, fontsize=8)


