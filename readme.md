# Image Classification with PyTorch

This project demonstrates **image classification** using a **Convolutional Neural Network (CNN)** built with PyTorch. The project utilizes the **Intel Image Dataset**, which contains images of various scenes (e.g., buildings, forests, glaciers) and demonstrates **data loading, augmentation, model training, and evaluation** techniques.

## **Table of Contents**
- [Dataset](#dataset)
- [Features](#features)
- [Training the Model](#training-the-model)
- [Results](#results)
- [References](#references)

---

## **Dataset**
The project uses the **Intel Image Classification Dataset**. This dataset contains images divided into six classes:
1. Buildings
2. Forests
3. Glaciers
4. Mountains
5. Seas
6. Streets

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) and place it in the following structure:
```plaintext
        datasets/
        └── intel_image_data/
            └── seg_train/
                └── seg_train/
                    ├── buildings/
                    ├── forest/
                    ├── glacier/
                    ├── mountain/
                    ├── sea/
                    └── street/

```

---

## **Features**
- **Data Augmentation**: Random rotations, horizontal flips, and color jittering to enhance training.
- **CNN Model**: A simple Convolutional Neural Network built with PyTorch.
- **Training and Validation**: PyTorch’s DataLoader is used to manage batches.
- **Model Saving**: Trained model is saved as `cnn_model.pth`.

---

## Training the Model
Here’s how the CNN model is trained:

-  Loss Function: CrossEntropyLoss is used for multi-class classification.
-  Optimizer: Adam optimizer with a learning rate of 0.001.
-  Epochs: Default training for 10 epochs (can be adjusted).
-  Batch Size: 32 images per batch.

---

## Results
Validation Accuracy: ~75.67% (after training for 10 epochs).

---

## References
- PyTorch Documentation
- Intel Image Classification Dataset on Kaggle
- Python TQDM – For displaying training progress bars.


"# Image_Augmentation_Pytorch" 
