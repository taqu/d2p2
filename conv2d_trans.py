import torch
import torch.nn as nn
import numpy as np

print('----1')
x = torch.from_numpy(np.random.rand(1,1,2,2).astype(np.float32))
conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
print('input:' + str(x))
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----2')
x = torch.from_numpy(np.random.rand(1,2,4,4).astype(np.float32))
conv = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True)
print('input:' + str(x))
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[[1,2],[3,4]]]], dtype=np.float32))
conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([[[[1, 2],[3, 4]]]], dtype=np.float32))
#print(str(conv.weight.data))
y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[
    [[1,2],[3,4]],
    [[1,2],[3,5]],
    [[1,2],[3,6]]
    ]], dtype=np.float32))
conv = nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=2, stride=2, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array(
    [
      [[[1,2],
       [3,4]],
      [[5,6],
       [7,8]]],
      [[[9,10],
       [11,12]],
      [[13,14],
       [15,16]]],
      [[[17,18],
       [19,20]],
      [[21,22],
       [23,24]]]
    ],dtype=np.float32))
#print(str(conv.weight.data))
y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

'''
print('--------------------------------------------------------')
x = torch.from_numpy(np.array(
    [[[[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12],
       [13,14,15,16]],
      [[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12],
       [13,14,15,16]]]],
    dtype=np.float32))
conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array(
    [[[[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]],
      [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]
    ]],
    dtype=np.float32))
y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))
'''

