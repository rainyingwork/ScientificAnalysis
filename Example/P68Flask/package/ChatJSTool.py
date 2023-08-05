
class ChatJSTool () :

    def getChartData (self) :
        return self.getExampleBarInfo()

    def getExampleBarInfo(self):
        return {
            "data": {
                "labels": ['2023/01', '2023/02','2023/03', '2023/04', '2023/05', '2023/06'],
                "datasets": [
                    {
                        "type": "bar",
                        "label": "A",
                        "data": [515, 533, 555, 501, 510, 489,],
                        "backgroundColor": 'rgba(75, 192, 192, 0.2)',
                        "borderColor": 'rgba(75, 192, 192, 1)',
                        "borderWidth": 1
                    },
                    {
                        "type": "bar",
                        "label": "B",
                        "data": [481, 321, 454, 421, 430, 544,],
                        "backgroundColor": 'rgba(1, 2, 3, 0.2)',
                        "borderColor": 'rgba(1, 2, 3, 1)',
                        "borderWidth": 1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "scales": {
                    "x": {
                        "stacked": False
                    },
                    "y": {
                        "stacked": False
                    }
                }
            }
        }

    def getExampleLineInfo(self):
        return {}

    def getExampleBubbleInfo(self):
        return {}

    def getExampleLineInfo(self):
        return {}

    def getExampleLineInfo(self):
        return {}

    def getExampleLineInfo(self):
        return {}