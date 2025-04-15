import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
import timm 
from senet_modul import SENetModule



def get_alexnet(output_size,  pretrained=False):

    alexnet = models.alexnet(pretrained=pretrained) 
    alexnet.classifier[5] = torch.nn.Dropout(0.5)  
    alexnet.classifier[6] = torch.nn.Linear(4096, output_size)  

    alexnet_resize = transforms.Resize((227, 227))


    return alexnet, alexnet_resize



def get_inceptionv3(output_size, pretrained=False):

    inceptionv3 = models.inception_v3(pretrained=pretrained, aux_logits=True)    
    inceptionv3.fc = torch.nn.Sequential( nn.Dropout(0.5), nn.Linear(2048, output_size) )

    inceptionv3_resize = transforms.Resize((299, 299))


    return inceptionv3, inceptionv3_resize



def get_vgg16(output_size, pretrained=False):

    vgg16 = models.vgg16(pretrained=pretrained)
    vgg16.classifier[6] = nn.Sequential( nn.Dropout(0.5),  nn.Linear(in_features=4096, out_features=output_size) )

    vgg16_resize = transforms.Resize((224, 224))


    return vgg16, vgg16_resize



def get_mobilenet(output_size, pretrained=False):

    mobilenet = timm.create_model('mobilenetv1_100', pretrained=pretrained)
    in_features = mobilenet.get_classifier().in_features  
    mobilenet.classifier = nn.Sequential( nn.Dropout(p=0.5), nn.Linear(in_features, output_size) )

    mobilenet_resize = transforms.Resize((224, 224))


    return mobilenet, mobilenet_resize



def get_resnet50(output_size, pretrained=False):

    resnet50 = models.resnet50(pretrained=pretrained)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential( nn.Dropout(0.5), nn.Linear(num_features, output_size) )

    resnet50_resize = transforms.Resize((224, 224))


    return resnet50, resnet50_resize



def get_mobilenet_senet(output_size, pretrained=True):

    mobilenet = timm.create_model('mobilenetv1_100', pretrained=pretrained, features_only=True)  
    in_features = mobilenet.feature_info.channels()[-1]  

    model = nn.Sequential(
        mobilenet,  
        SENetModule(in_features),  
        nn.AdaptiveAvgPool2d(1), 
        nn.Flatten(),
        nn.Dropout(0.5),  
        nn.Linear(in_features, output_size)  
    )

    mobilenet_resize = transforms.Resize((224, 224))


    return model, mobilenet_resize



