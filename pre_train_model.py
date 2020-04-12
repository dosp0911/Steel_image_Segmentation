import torchvision.models as models

pre_vgg = models.vgg16_bn(pretrained=True)

for name, param in pre_vgg.named_parameters():
    print(name, param)

pre_vgg