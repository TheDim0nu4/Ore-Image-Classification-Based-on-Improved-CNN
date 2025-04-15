from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import architectures as arch
import os



def testing(model_filepath, test_dataset_path):

    models_dict = {"alexnet": arch.get_alexnet(7), "inceptionv3": arch.get_inceptionv3(7), "vgg16": arch.get_vgg16(7),  "resnet50": arch.get_resnet50(7), "mobilenet_senet": arch.get_mobilenet_senet(7), "mobilenet": arch.get_mobilenet(7)}

    for key, value in models_dict.items():

        if key in model_filepath:

            model = value[0]
            model.load_state_dict( torch.load(model_filepath), strict=False )

            resize = value[1]

            break



    if "_DA" in model_filepath:
        stats = torch.load("normalizations/normals_stats_DA.pt")
    else:
        stats = torch.load("normalizations/normals_stats.pt")

    mean, std = stats['mean'], stats['std']

            
    transform = transforms.Compose([transforms.ToTensor(), resize, transforms.Normalize(mean=mean.tolist(), std=std.tolist())])
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True) 



    model.eval()
    correct_test, total_test = 0, 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)


    test_accuracy = correct_test / total_test


    return test_accuracy




if __name__ == "__main__":

    folder_path = "models"
    file_names = os.listdir(folder_path)

    result = []

    for file_name in file_names:

        model_filepath = f"models/{file_name}"
        accuracy = testing(model_filepath, "testing_dataset/")

        result.append((file_name, accuracy))


    for name, res in result:

        print(f"Model: {name.replace('.pth', '')}. Accuracy: {res}")




 



