import torch
import torch.nn as nn
import numpy as np

print('----1')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----2')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----3')
x = torch.from_numpy(np.random.rand(1,2,3).astype(np.float32))
conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----4')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----5')
x = torch.from_numpy(np.random.rand(1,2,3).astype(np.float32))
conv = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----6')
conv = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----7')
conv = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=6, stride=1, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
x = torch.from_numpy(np.random.rand(4,2,6).astype(np.float32))
y = conv(x)
print('result shape:' + str(y.shape))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2,3],[4,5,6]]], dtype=np.float32))
conv = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([
    [[1, 2, 3],[4, 5, 6]],
    [[1, 2, 3],[4, 5, 7]],
    [[1, 2, 3],[4, 5, 8]]], dtype=np.float32))
#print(str(conv.weight.data))

y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2,3,4,5,6],[1,2,3,4,5,6]],[[7,8,9,10,11,12],[1,2,3,4,5,6]],[[13,14,15,16,17,18],[1,2,3,4,5,6]],[[19,20,21,22,23,24],[1,2,3,4,5,6]]], dtype=np.float32))
conv = nn.Conv1d(in_channels=2, out_channels=5, kernel_size=6, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([
    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],

    [[3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
     [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]],

    [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
     [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]],

    [[7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
     [8.0, 8.0, 8.0, 8.0, 8.0, 8.0]],

    [[9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
     [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
    ], dtype=np.float32))
y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

