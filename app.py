from flask import Flask, request, jsonify
from torch import FloatTensor
from load import model
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.data import TabularDataset
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

'''
Web => AI 보내는거
Request:
    body:

    1)include
     사용자가 선택한 좋아하는 음식의 index 값의 리스트
     ex) [1,3,5,7,9,]  // 12개

    2)exclude
     사용자가 선택한 싫어하는 음식의 이름 리스트
     ex)["가지", "생선"]

AI=> Web으로 보내줘야하는거
Response:
    body:
        food: []
        //추천 음식 리스트 (6개)
'''


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(309, 200),
            nn.Linear(200, 100),
            nn.ReLU(True),  # 이건 왜쓴거지? 모르겠음
            nn.Linear(100, 1))
        self.decoder = nn.Sequential(
            nn.Linear(1, 100),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 309), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

for epoch in range(num_epochs):
    for data in dataloader:
        output = model(data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def AIModel(include, exclude):
    num_epochs = 150
    batch_size = 50

    learning_rate = 1e-3

    t1_data = pd.read_csv('./11/08.csv')

    nb_users = int(max(t1_data.iloc[:, 0])) + 1  # 사용자의 수 +1
    nb_foods = int(max(t1_data.iloc[:, 1])) + 1  # 음식 종류의 수 만약 인덱스 0부터 주었다면 +1

    t1_data = t1_data.values

    def convert(data):
        new_data = []
        for id_users in range(0, nb_users):  # 총 사용자 수많큼 반복해라
            id_foods = data[:, 1][data[:, 0] == id_users]  # user가 본 영화들
            id_foods = id_foods.astype(int)  # user가 본 영화들의 별점
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = np.zeros(nb_foods)  # 영화 숫자만큼 zero 배열 만들어줌
            ratings[
                id_foods] = id_ratings  # id_movies영화갯수 1부터 하려고 -1을 해줌/ id_movies - 1번째 영화 /ratings[id_movies - 1]: n번째 영화 별점이 몇점인지 쭉 나열
            ratings = ratings.astype(float)
            new_data.append(list(ratings))  # 전체영화 zero에 배열되있는것에 점수 넣어줌

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

        return new_data

    t2_data = convert(t1_data)
    t2_data = np.asarray(t2_data)

    tensor = torch.FloatTensor(t2_data)

    num_train_dataset = int(len(tensor) * 0.8)
    num_test_dataset = len(tensor) - num_train_dataset

    train_dataset, test_dataset = torch.utils.data.random_split(tensor, [num_train_dataset, num_test_dataset])

    dataloader = DataLoader(tensor, batch_size=batch_size, shuffle=True)

    aaa = []
    lists = include
    print(lists)
    arr = [0 for i in range(1, 310)]
    count = 0
    for i in lists:
        if count < 4:
            arr[i] = 5
        elif count < 8:
            arr[i] = 3
        elif count < 12:
            arr[i] = 1
        count += 1

    aaa.append(arr)
    bb = torch.FloatTensor(aaa)

    new_user_input = bb
    output = model(new_user_input)

    output = (output + 1)

    sort_food_id = np.argsort(-output.detach().numpy())
    sort_food_id_list = sort_food_id.tolist()
    food_real_list = np.ravel(sort_food_id_list, order='C').tolist()

    file = pd.read_excel('food_label.xlsx')
    rm_list = set()
    list_remove = exclude

    for j in list_remove:
        for i in range(309):
            if file[j][i] == 1:
                rm_list.add(file['f_num'][i])

    rm_real_list = list(rm_list)

    for i in food_real_list:
        if i in rm_list or i in include:
            food_real_list.remove(i)

    top_10 = food_real_list[:10]

    count1 = 2
    sampleList1 = include
    random_list1 = random.sample(sampleList1, count1)

    count = 4
    sampleList = top_10
    random_list2 = random.sample(sampleList, count)

    final_list = random_list1 + random_list2

    return final_list


@app.route('/', methods=['POST'])
def AiModel():
    include = request.json['include']  # 좋아하는 음식 (인덱스 리스트)
    exclude = request.json['exclude']  # 싫어하는 음식 (이름 리스트)

    data = AIModel(include, exclude)

    return jsonify(data)


if __name__ == "__main__":
    app.run(port=5000)