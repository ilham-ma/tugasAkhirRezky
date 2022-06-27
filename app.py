from unittest import result
from flask import Flask
from flask import request
import json
from machine_learning.training import training
from machine_learning.model import run_model


app = Flask('app')

@app.route("/rezky", methods=['POST'])
def hello_world():
    response = json.loads(request.data)
    arrayStr = response['array'].split(',')
    arrayInt = list(map(int, arrayStr))
    result = run_model(X_train, X_test, y_train, y_test, arrayInt)

    return str(result)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = training()
    app.run('127.0.0.1', '5000')