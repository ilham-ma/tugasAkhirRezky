from unittest import result
from flask import Flask
from flask import request
import json
from machine_learning.training import training
from machine_learning.model import run_model


app = Flask('app')

@app.route("/", methods=['POST'])
def hello_world():
    response = json.loads(request.data)
    arrayStr = response['array'].split(',')
    arrayInt = list(map(int, arrayStr))
    result = run_model(X_train, X_test, y_train, y_test, arrayInt)

    return str(result)

@app.route("/")
def root():
    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = training()
    app.run(host="0.0.0.0", port=5000)
