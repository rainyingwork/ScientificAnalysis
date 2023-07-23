import os
from flask import Flask, render_template

app = Flask(__name__, template_folder='file/html' , static_folder='file/html')

@app.route('/')
def index():
    # Render the template with the chart
    return render_template('temp.html')

@app.route('/RestAPI')
def index_rest():
    # Render the template with the chart
    return {'test':123456}

if __name__ == '__main__':
    app.run(debug=True,port=5050)