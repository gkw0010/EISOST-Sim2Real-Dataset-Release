# loss板块
import torch.nn as nn
import torch

class Content_Loss(nn.Module):#内容损失
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()#继承父类的初始化
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()#利用均方误差计算损失
        
    def forward(self, input):#向前计算损失
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out
        
    def backward(self, retain_graph=True):#反向求导
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
        
        
class Gram(nn.Module):#定义Gram矩阵
    def __init__(self):
        super(Gram, self).__init__()
        
    def forward(self, input):#向前计算Gram矩阵
        a, b, c, d = input.size()#a为批量大小，b为feature map的数量，c*d为feature map的大小
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram
        
        
class Style_Loss(nn.Module):#风格损失
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out
        
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss