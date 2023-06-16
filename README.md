# Garbage Classification Project

In line with our goals, the primary aim of the project is to promote waste sorting awareness, particularly among children. By targeting this specific age group, we believe that instilling the knowledge and practices of responsible waste management from an early age will lay the foundation for a cleaner and more sustainable future. Our project aspires to empower children to become proactive agents of change in their communities.

# Dataset

The dataset used for training and testing the model is sourced from Kaggle.com. It consists of labeled images of various garbage items, categorized into the three classes mentioned above.

You can find the dataset used in this project at [kaggle.com](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification). Make sure to download and organize the dataset into the appropriate folders (train, valid, test) before running the code.

# Model

The classification model is based on the MobileNetV2 architecture, a lightweight and efficient CNN model. Transfer learning is employed using pre-trained weights from the ImageNet dataset. Data augmentation techniques like rotation, shifting, and flipping are used during training.

# Usage

To use this project, follow these steps:

    Find and download the dataset you want to train.
    Train your model using the provided code.
    Test the trained model.

To start training your model, run the following command:

``` bash
python train.py
```
To test the model, use the following command:

```bash
python test.py
```
Remember, more data leads to better results. Consider expanding your dataset or collecting more samples to improve the model's performance.

# Note

If you encounter any issues or have trouble finding a suitable dataset, visit [kaggle.com](https://www.kaggle.com). It's a popular platform for machine learning datasets and competitions, offering a wide range of datasets.

Feel free to explore and modify the code to improve the model's performance or adapt it to your specific needs.

This revised note provides a concise overview of the project, focusing on the essential information while maintaining clarity.
