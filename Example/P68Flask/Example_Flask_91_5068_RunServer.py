import os
from flask import Flask, render_template

app = Flask(__name__, template_folder='file/html' , static_folder='file/html')

@app.route('/')
def index():
    # Render the template with the chart
    return render_template('main.html')


@app.route('/getChatData')
def chart_data_1():
    import sys
    try:
        del sys.modules['Example.P68Flask.package.ChatJSTool']
    except:
        pass
    from Example.P68Flask.package.ChatJSTool import ChatJSTool
    chatJSTool = ChatJSTool()
    return chatJSTool.getChartData()

@app.route('/getButtonTypes')
def getBottonTypes():
    buttonMap = {
        'button01':'按鈕01','button02':'按鈕02'
    }
    buttonTypes = []
    for i, button in enumerate(buttonMap):
        buttonTypes.append({ "id": "open-button-selection-form-{}".format(button), "button": button, "buttonName": buttonMap[button] })
    return buttonTypes

@app.route('/getButtonOptions/<buttonName>')
def getBottonOptions(buttonName):
    print()
    if buttonName == 'button01':
        return {'選項1': 1, '選項2': 2, '選項3': 3, '選項4': 4 , '選項5':5}
    elif buttonName == 'button02':
        return {'選項A': 'A', '選項B': 'B', '選項C': 'C', '選項D': 'D'}
    else:
        return 'Invalid button name'

if __name__ == '__main__':
    app.run(debug=True,port=5068)