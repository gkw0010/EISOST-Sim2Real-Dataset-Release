# load_img模块
import PIL.Image as Image
import torch
import torchvision.transforms as transforms

img_size = 512 if torch.cuda.is_available() else 128#根据设备选择改变后项数大小
def load_img(img_path):#图像读入
    img = Image.open(img_path).convert('RGB')#将图像读入并转换成RGB形式
    w,h = img.size[0], img.size[1]
    img = img.resize((img_size, img_size))#调整读入图像像素大小
    img = transforms.ToTensor()(img)#将图像转化为tensor
    img = img.unsqueeze(0)#在0维上增加一个维度
    return img, w, h

def show_img(img):#图像输出
    img = img.squeeze(0)#将多余的0维通道删去
    img = transforms.ToPILImage()(img)#将tensor转化为图像
    img.show()