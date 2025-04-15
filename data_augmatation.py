import os
import cv2
import numpy as np
from tqdm import tqdm



def create_augmatation_dataset(dataset_path, augmented_dataset_path, augmentations):

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        new_class_path = os.path.join(augmented_dataset_path, class_name)
        os.makedirs(new_class_path, exist_ok=True)


        for image_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, image_name)
            img = cv2.imread(img_path)

            if img is None:
                continue


            cv2.imwrite(os.path.join(new_class_path, image_name), img)


            for aug_name, aug_func in augmentations.items():
                augmented_img = aug_func(img)
            

                if augmented_img.shape[0] > 0 and augmented_img.shape[1] > 0:
                    aug_image_name = f"{os.path.splitext(image_name)[0]}_{aug_name}.jpg"
                    cv2.imwrite(os.path.join(new_class_path, aug_image_name), augmented_img)





if  __name__ == "__main__":

    dataset_path = "dataset"
    augmented_dataset_path = "dataset_augmented"

    augmentations = {
        "flipped": lambda img: cv2.flip(img, 1),  
        "zoom_left": lambda img: img[:, :int(0.8 * img.shape[1])],  
        "zoom_right": lambda img: img[:, int(0.2 * img.shape[1]):],  
        "darkened": lambda img: np.clip(img * 0.5, 0, 255).astype(np.uint8),  
    }


    create_augmatation_dataset(dataset_path, augmented_dataset_path, augmentations)




