import torch
import torch.nn as nn
import numpy as np

print('----1')
x = torch.from_numpy(np.random.rand(1,1,1).astype(np.float32))
conv = nn.Linear(in_features=1, out_features=1, bias=True)
print('weight shape:' + str(conv.weight.shape))
print('bias shape:'+ str(conv.bias.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----2')
x = torch.from_numpy(np.random.rand(2,1,3).astype(np.float32))
conv = nn.Linear(in_features=3, out_features=2, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('----3')
x = torch.from_numpy(np.random.rand(1,2,2).astype(np.float32))
conv = nn.Linear(in_features=2, out_features=3, bias=False)
print('weight shape:' + str(conv.weight.shape))
y = conv(x)
print('result shape:' + str(y.shape))

print('--------------------------------------------------------')
x = torch.from_numpy(np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]], dtype=np.float32))
conv = nn.Linear(in_features=2, out_features=3, bias=False)

print('input:' + str(x.shape))
print('input:' + str(x))

print('weight shape:' + str(conv.weight.shape))
#print('weight:' + str(conv.weight.data))
conv.weight.data = torch.from_numpy(np.array([
    [1, 2],[3,4],[5,6]],
    dtype=np.float32))
#print(str(conv.weight.data))

y = conv(x)
print('result:' + str(y.shape))
print('result:' + str(y))

