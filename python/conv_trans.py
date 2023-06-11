import torch
import torch.nn as nn
import numpy as np

print('----1')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----2')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----3')
x = torch.from_numpy(np.random.rand(1,2,3).astype(np.float32))
conv = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----4')
x = torch.from_numpy(np.random.rand(1,1,3).astype(np.float32))
conv = nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----5')
x = torch.from_numpy(np.random.rand(1,2,3).astype(np.float32))
conv = nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----6')
conv = nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----7')
conv = nn.ConvTranspose1d(in_channels=2, out_channels=5, kernel_size=6, stride=1, padding=0, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
x = torch.from_numpy(np.random.rand(4,2,6).astype(np.float32))
y = conv(x)
print('result shape:' + str(y.shape))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2,3],[4,5,6]]], dtype=np.float32))
conv = nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([
    [[1, 2, 3],[4, 5, 6],[7,8,9]],
    [[1, 2, 3],[4, 5, 6],[7,8,10]]
    ], dtype=np.float32))
print(str(conv.weight.data))

y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2,3],[4,5,6]]], dtype=np.float32))
conv = nn.ConvTranspose1d(in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=0, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print(str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([
    [[1, 2],[3, 4],[5, 6]],
    [[1, 2],[3, 4],[5, 7]]
    ], dtype=np.float32))
#print(str(conv.weight.data))

y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))


'''
print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2,3,4,5,6]]], dtype=np.float32))
conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

print(x.shape)
print(x)

print(conv.weight.shape)
conv.weight.data = torch.from_numpy(np.array([[[1, 2, 3]]], dtype=np.float32))
print(conv.weight.data)

y = conv(x)
print(y.shape)
print(y)

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2]],[[3,4]],[[5,6]]], dtype=np.float32))
conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

print(x.shape)
print(x)

print('weight.shape: ' + str(conv.weight.shape))
conv.weight.data = torch.from_numpy(np.array([[[1, 2, 3]]], dtype=np.float32))

y = conv(x)
print(y.shape)
print(y)
'''

