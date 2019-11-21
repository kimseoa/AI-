from flask import Flask, request, jsonify
from torch import FloatTensor
from load import model

app = Flask(__name__)

'''
Request:
    body:
        foods: number[]

Response:
    body:
        recommendation: number[]
'''

@app.route('/', methods=['POST'])
def AiModel():
    data= request.json['foods']
    print(type(data)) #list
    #이부분은 파이토치? 로 가공해서 리스트로 주세요
    # data =  ....
    return jsonify(data)


# @app.route('/', methods=['POST'])
# def AiModel():
#     body = request.get_json(force=True)
#     output = model(FloatTensor(body['foods'])).detach().numpy()


if __name__ == "__main__":
    app.run(port=5000)
    # app.run(debug=True, port=5000)
