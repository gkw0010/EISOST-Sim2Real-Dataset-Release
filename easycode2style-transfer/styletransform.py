# maincode
import os
from torch.autograd import Variable
from torchvision import transforms
import torch
from run_code import run_style_transfer
from load_img import load_img
import os, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_root = '/mnt/data-hdd/wgk/mmsegmentation/data/images_gap/images/realdata/'   # style images root
root = '/mnt/data-hdd/wgk/mmsegmentation/data/images_gap/images/training/'       # content images root
content_imgs = os.listdir(root)
style_imgs = os.listdir(style_root)
number = 0
for i in range(len(content_imgs)):
    style_img, w, h = load_img(style_root + random.sample(style_imgs, int(1))[0])
    style_img = Variable(style_img).to(device)
    img = content_imgs[i]
    number = number + 1
    print(number)
    content_img, w, h = load_img(root + img)
    content_img = Variable(content_img).to(device)
    input_img = content_img.clone()

    out = run_style_transfer(content_img, style_img, input_img, num_epoches=150)
    save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
    save_pic = save_pic.resize((w, h))
    save_pic.save(root.replace('training', 'training_st') + img)