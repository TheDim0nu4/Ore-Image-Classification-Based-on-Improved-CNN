from torch.utils.data import DataLoader, random_split
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torch
import architectures as arch



def train(model, resize, dataset_path, save_train_result_path, save_model_path):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    print(f"Using device: {device}") 


    def calculate_mean_std(dataset):

        mean = torch.zeros(3, device=device)  
        std = torch.zeros(3, device=device)

        for images, _ in dataset:

            images = images.to(device)
            mean += torch.mean(images, dim=[1, 2]) 
            std += torch.std(images, dim=[1, 2])  
         
        mean /= len(dataset)
        std /= len(dataset)

        return mean.cpu(), std.cpu() 


    transform = transforms.Compose([
        resize,  
        transforms.ToTensor()           
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    mean, std = calculate_mean_std(dataset)




    transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist()) ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    train_size = int(0.6 * len(dataset))
    test_size =  int((len(dataset) - train_size)/2)
    validation_size = len(dataset) - (train_size+test_size)


    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)


    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, {labels.shape}\n")



    model.to(device)

    epochs = 50
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss() 


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracy, validation_accuracy = [], []


    for epoch in range(epochs):

        model.train()  
        correct_train, total_train = 0, 0

        for inputs, labels in train_loader:  

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step() 

            with torch.no_grad():

                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

        train_accuracy.append( correct_train / total_train )


        correct_validation, total_validation = 0, 0

        with torch.no_grad():

            for inputs, labels in validation_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, predicted = torch.max(outputs, 1)
                correct_validation += (predicted == labels).sum().item()
                total_validation += labels.size(0)

        validation_accuracy.append( correct_validation / total_validation )


        print(f"Epoch: {epoch}. Validation accuracy: {correct_validation / total_validation}")




    model.eval()
    correct_test, total_test = 0, 0


    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)


    test_accuracy = correct_test / total_test


    print(f"Test accuracy: {test_accuracy}")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {validation_accuracy}")





    torch.save(model.state_dict(), save_model_path)

    with open(save_train_result_path, 'w') as f:
        f.write(f"{test_accuracy}\n")
        f.write(f"{train_accuracy}\n")
        f.write(f"{validation_accuracy}\n")






if __name__ == "__main__":

    output_size = 7


    # Simple models

    model, resize = arch.get_alexnet(output_size)
    train(model, resize, "dataset/", "result_training/alexnet.txt", "models/alexnet.pth")

    model, resize = arch.get_inceptionv3(output_size)
    train(model, resize, "dataset/", "result_training/inceptionv3.txt", "models/inceptionv3.pth")

    model, resize = arch.get_vgg16(output_size)
    train(model, resize, "dataset/", "result_training/vgg16.txt", "models/vgg16.pth")

    model, resize = arch.get_resnet50(output_size)
    train(model, resize, "dataset/", "result_training/resnet50.txt", "models/resnet50.pth")

    model, resize = arch.get_mobilenet(output_size)
    train(model, resize, "dataset/", "result_training/mobilenet.txt", "models/mobilenet.pth")



    # With transfer learning

    model, resize = arch.get_inceptionv3(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/inceptionv3_TL.txt", "models/inceptionv3_TL.pth")

    model, resize = arch.get_vgg16(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/vgg16_TL.txt", "models/vgg16_TL.pth")

    model, resize = arch.get_resnet50(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/resnet50_TL.txt", "models/resnet50_TL.pth")

    model, resize = arch.get_mobilenet(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/mobilenet_TL.txt", "models/mobilenet_TL.pth")



    # With transfer learning and data augmatation

    model, resize = arch.get_inceptionv3(output_size, pretrained=True)
    train(model, resize, "dataset_augmented/", "result_training/inceptionv3_TL_DA.txt", "models/inceptionv3_TL_DA.pth")

    model, resize = arch.get_vgg16(output_size, pretrained=True)
    train(model, resize, "dataset_augmented/", "result_training/vgg16_TL_DA.txt", "models/vgg16_TL_DA.pth")

    model, resize = arch.get_resnet50(output_size, pretrained=True)
    train(model, resize, "dataset_augmented/", "result_training/resnet50_TL_DA.txt", "models/resnet50_TL_DA.pth")

    model, resize = arch.get_mobilenet(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/mobilenet_TL_DA.txt", "models/mobilenet_TL_DA.pth")



    # Mobilenet with senet module and TL, DA

    model, resize = arch.get_mobilenet_senet(output_size, pretrained=True)
    train(model, resize, "dataset/", "result_training/mobilenet_senet_TL_DA.txt", "models/mobilenet_TL_DA.pth")

  
    




