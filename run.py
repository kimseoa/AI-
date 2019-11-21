from load import model
import torch
import numpy as np

new_user_input = torch.FloatTensor([0, 0, 0, 3, 1, 2])
output = model(new_user_input)

output = (output + 1)
print(output)

# 주어진 테이블
food_table = []

print("모델 Output")
print(output)

print("최고 별점 음식 추천")
print(np.sort(output.detach().numpy()))

print("최고 별점의 food id 리스트")
sort_food_id = np.argmax(output.detach().numpy())
print(sort_food_id)
