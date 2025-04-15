import matplotlib.pyplot as plt



def vizual_training(filepath, graph_name):


    with open(filepath, 'r') as file:
        lines = file.readlines()
        test_accuracy = float(lines[0].strip())  
        train_accuracy = eval(lines[1].strip())  
        validation_accuracy = eval(lines[2].strip())  


    epochs = 50

    plt.figure(figsize=(8,6))


    plt.scatter(epochs, test_accuracy, color='red', label=f'Test accuracy: {test_accuracy:.4f}', s=100)


    plt.plot(range(1, epochs+1), train_accuracy, color='green', marker='o', markersize=5, label='Training accuracy', alpha=0.8)
    plt.plot(range(1, epochs+1), validation_accuracy, color='blue', marker='o', markersize=5, label='Validation accuracy', alpha=0.8)


    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{graph_name}')
    plt.legend()


    plt.grid(True)
    plt.show()




def vizual_training_with_TL_DA(filepath_TL, filepath_TL_DA, graph_name):


    with open(filepath_TL, 'r') as file:
        lines = file.readlines()
        test_accuracy_TL = float(lines[0].strip())  
        train_accuracy_TL = eval(lines[1].strip())  
        validation_accuracy_TL = eval(lines[2].strip())  


    with open(filepath_TL_DA, 'r') as file:
        lines = file.readlines()
        test_accuracy_TL_DA = float(lines[0].strip())  
        train_accuracy_TL_DA = eval(lines[1].strip())  
        validation_accuracy_TL_DA = eval(lines[2].strip())  


    epochs = 50

    plt.figure(figsize=(8,6))


    plt.scatter(epochs, test_accuracy_TL, color='yellow', label=f'Test accuracy with TL: {test_accuracy_TL:.4f}', s=100)
    plt.plot(range(1, epochs+1), train_accuracy_TL, color='orange', marker='o', markersize=5, label='Training accuracy with TL', alpha=0.8)
    plt.plot(range(1, epochs+1), validation_accuracy_TL, color='purple', marker='o', markersize=5, label='Validation accuracy with TL', alpha=0.8)


    plt.scatter(epochs, test_accuracy_TL_DA, color='red', label=f'Test accuracy with TL and DA: {test_accuracy_TL_DA:.4f}', s=100)
    plt.plot(range(1, epochs+1), train_accuracy_TL_DA, color='green', marker='o', markersize=5, label='Training accuracy with TL and DA', alpha=0.8)
    plt.plot(range(1, epochs+1), validation_accuracy_TL_DA, color='blue', marker='o', markersize=5, label='Validation accuracy with TL and DA', alpha=0.8)



    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{graph_name}')
    plt.legend()


    plt.grid(True)
    plt.show()





if __name__ == "__main__":

    models = ["AlexNet", "InceptionV3", "Vgg16", "MobileNet", "ResNet50"]  


    for model_name in models:   
        vizual_training( f"result_training/{model_name.lower()}.txt", model_name)


    for model_name in models[1:]:
    
        filepath_TL = f"result_training/{model_name.lower()}_TL.txt"
        filepath_TL_DA = f"result_training/{model_name.lower()}_TL_DA.txt"

        vizual_training_with_TL_DA(filepath_TL, filepath_TL_DA, model_name)





