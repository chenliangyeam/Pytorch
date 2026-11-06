import torch
import torch.nn as nn
import pandas as pd

raw_df = pd.read_csv('train.csv')

#標籤
label = raw_df['label'].values

#特徵
#把標籤刪掉就可以獲得特徵
raw_df = raw_df.drop(['label'], axis = 1)
feature = raw_df.values

#把資料分成train data和test data
train_feature = feature[:int(len(feature)*0.8)]
train_label = label[:int(len(label)*0.8)]
test_feature = feature[int(len(feature)*0.8):]
test_label = label[int(len(label)*0.8):]

train_feature = torch.tensor(train_feature).to(torch.float)
train_label = torch.tensor(train_label)
test_feature = torch.tensor(test_feature).to(torch.float)
test_label = torch.tensor(test_label)

#GPU加速
# train_feature = torch.tensor(train_feature).to(torch.float).cuda()
# train_label = torch.tensor(train_label).cuda()
# test_feature = torch.tensor(test_feature).to(torch.float).cuda()
# test_label = torch.tensor(test_label).cuda()

#784個像素點構成的灰度圖->函數->10個概率

#輸入層 in_channel 784 out_channel 444
#隱藏層1 in_channel 444 out_channel 512
#隱藏層2 in_channel 512 out_channel 512
#輸出層 in_channel 512 out_channel 10

model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),  #激活函數
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),  #結果會是10個正無限到負無限的值
    nn.Softmax()
)
#GPU加速
#model = model.cuda()

#梯度下降
loss_function = nn.CrossEntropyLoss()

#優化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for i in range(100):
    #train
    optimizer.zero_grad()  #清空優化器梯度
    predict = model(train_feature)

    #predict的結果會是一個10行多列的矩陣
    #0.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.0 -> 0
    #0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.9 0.1 0.0 -> 7
    #0.0 0.0 0.0 0.0 0.0 0.9 0.0 0.0 0.1 0.0 -> 5
    #0.0 0.0 0.0 0.9 0.0 0.0 0.0 0.0 0.1 0.0 -> 3

    result = torch.argmax(predict, axis=1)  #取出每一raw最大值的索引
    train_acc = torch.mean((result==train_label).to(torch.float))  #計算準確率

    loss = loss_function(predict, train_label)  #計算損失
    loss.backward()  #反向傳播
    optimizer.step()  #更新權重
    print('train loss:{}, train acc:{}'.format(loss.item(), train_acc.item()))

    #test
    optimizer.zero_grad()  #清空優化器梯度
    predict = model(test_feature)
    result = torch.argmax(predict, axis=1)  #取出每一raw最大值的索引
    test_acc = torch.mean((result==test_label).to(torch.float))  #計算準確率

    loss = loss_function(predict, test_label)  #計算損失
    print('test loss:{}, test acc:{}'.format(loss.item(), test_acc.item()))

torch.save(model.state_dict(), './model.pt')  #儲存模型權重

#加載模型
#params = torch.load('./model.pt')
#model.load_state_dict(params)