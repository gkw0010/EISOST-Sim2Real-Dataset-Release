# build_model模块
import torch.nn as nn
import torch
import torchvision.models as models
import loss  # 指的是上文中已经写好的loss模块

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#选择运行设备，如果你的电脑有gpu就在gpu上运行，否则在cpu上运行
vgg = models.vgg19(pretrained=True).features.to(device)#这里我们使用预训练好的vgg19模型
'''所需的深度层来计算风格/内容损失:'''
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    content_loss_list = [] #内容损失
    style_loss_list = [] #风格损失
    model = nn.Sequential() #创建一个model，按顺序放入layer
    model = model.to(device)
    gram = loss.Gram().to(device)
    
    '''把vgg19中的layer、content_loss以及style_loss按顺序加入到model中：'''
    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)
            if name in content_layers_default:
                target = model(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)
            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = loss.Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)
            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)
        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)
            
    return model, style_loss_list, content_loss_list