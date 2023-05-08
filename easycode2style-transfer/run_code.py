# run_code模块
import torch.nn as nn
import torch.optim as optim
from build_model import get_style_model_and_loss
import os

def get_input_param_optimier(input_img):
    """input_img is a Variable"""
    input_param = nn.Parameter(input_img.data)#获取参数
    optimizer = optim.LBFGS([input_param])#用LBFGS优化参数
    return input_param, optimizer
    
def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)
    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:#每隔50次输出一次loss
        def closure():
            input_param.data.clamp_(0, 1)#修正输入图像的值
            model(input_param)
            style_score = 0
            content_score = 0
            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
        input_param.data.clamp_(0, 1)#再次修正
    return input_param.data