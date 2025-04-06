# Model Performance Summary

## **Inference Accuracy**
| Model          | Accuracy |
|---------------|----------|
| **AlexNet**   | 17.81%   |
| **ResNet18**  | 52.89%   |
| **EfficientNet** | 17.78% |
| **VGG16**     | 17.46%   |
| **InceptionV3**  |  28.32% |

---

## **Model Performance with Imbalanced Dataset**

| Model          | Train Accuracy | Validation Accuracy| Test Accuracy |
|---------------|----------|----------|------------|
| **AlexNet**   | 52.89%   | 52.75% | 52.99% |
| **ResNet18**  | 62.17%   | 43.04% | 42.16% |
| **EfficientNetb3** | 83.28% | 59.87% | 58.80% |  
| **VGG16**     |  52.89%  | 52.75% | 52.99% | 
| **InceptionV3** |  51.78%  | 47.25% | 47.98% | 

---

## **Training Details (ResNet50 - With UnderSampling)**
| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|-----------|---------------|----------|-------------|
| 1     | 1.1471    | 37.28%        | 1.1009   | 30.69%      |
| 2     | 1.0833    | 41.61%        | 1.5384   | 23.42%      |
| 3     | 1.0646    | 42.15%        | 1.3209   | 51.05%      |
| 4     | 1.0321    | 45.55%        | 1.1372   | 37.00%      |
| 5     | 1.0025    | 48.11%        | 2.7209   | 26.33%      |
| 6     | 0.9821    | 50.35%        | 1.5455   | 29.89%      |
| 7     | 0.9565    | 53.91%        | 1.1063   | 40.39%      |
| 8     | 0.9325    | 53.75%        | 1.6276   | 20.36%      |
| 9     | 0.8946    | 57.00%        | 1.8916   | 27.14%      |
| 10    | 0.8385    | 61.87%        | 1.6393   | 32.96%      |

---

## **Observations**
1. **AlexNet, EfficientNet, and VGG16 showed poor inference accuracy (~17-18%)**, indicating they struggle with the dataset.
2. **ResNet18 achieved the highest inference accuracy (52.89%)**, suggesting it is better suited for the task.
3. **VGG16 had better test accuracy (52.99%)** compared to ResNet18 (42.16%), despite poor inference accuracy.
4. **ResNet50 training showed improvement in accuracy over epochs, but validation accuracy fluctuated** and dropped in later epochs.
5. **Overfitting might be an issue in ResNet50**, as validation loss and accuracy do not consistently improve.

---

## **Next Steps**
- Investigate **dataset balancing strategies** further.
- Consider **data augmentation** to improve generalization.
- Experiment with **learning rate schedules** to stabilize training.
- **Fine-tune pre-trained models** on the dataset instead of training from scratch.

---

### **Conclusion**
ResNet18 and VGG16 seem to perform best, but dataset imbalance and overfitting need further investigation to improve generalization.

## Model Training and Testing Conducted via 5-Fold Cross Validation

## Model Performance for augmented-images-v1 (horizontal flip and color jitter)

| **Model**     | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** | **Epochs** | **Learing Rate** |
|---------------|----------------|--------------------|--------------|------------------|------------|------------------|
| ResNet50      | 0.9288         | 54.69%             | 0.9687       | 48.79%           | 10         |        NA        |
| VGGNET        | 0.0139         | 99.75%             | 1.5022       | 72.91%           | 20         |        NA        |
| AlexNet       | 0 .4233        | 81 .58%            | 0.8465       | 67.72%           | 20         |        NA        |
| InceptionV3   | 0.0134         | 99.72%             | 0.9121       | 73.52%           | 20         |        NA        |
| EffNetB3      | 0.7023         | 67.14%             | 0.7755       | 62.93%           | 20         |        NA        |
| EffNetB3      | 0.8974         | 55.86%             | 0.8681       | 56.72%           | 10         | 0.0001           |
| ResNet18      | 0.0214         | 99.59%             | 1.0446       | 71.79%           | 10         | 0.001            |

## Model Performance for augmented-images-v2 (horizontal flip and vertical flip)

| **Model**     | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** | **Epochs** | **Learing Rate** |
|---------------|----------------|--------------------|--------------|------------------|------------|------------------|
| ViT           | 0.0065         | 99.80%             | 2.8423       | 53.77%           | 25         | 0.001            |
| InceptionV3   | 0.1235         | 97.43%             | 0.9228       | 63.03%           | 15         | 0.0001           |

## Model Performance for augmented-images-v3 (horizontal flips, 900 ~ samples per class)

| Model         | Average Training Accuracy | Average Validation Accuracy       | Epochs Per Fold          | **Learning Rate** |
|---------------|-------------------|-----------------------------------|-------------------|-------------------|
| InceptionV3   | 99.00%            | 71.17%                            | 30                | 0.001             |
| InceptionV3   | 92.18%            | 56.78%                            | 15                | 0.0001            |
| EffcientNetB3 | 79.65%            | 59.86%                            | 10                | 0.001             |
| AlextNet      | 89.97%            | 59.59%                            | 30                | 0.001             |
| ViT           | 93.89%            | 46.47%                            | 25                | 0.001             | 
| MobileNet_V2  | 99.59%            | 62.37%                            | 15                | 0.001             |
| DenseNet-121  | 99.77%            | 71.07%                            | 15                | 0.001             |
| ResNet18      | 99.99%            | 67.91%                            | 30                | 0.0001            |
| ResNet18      | 100%              | 64.92%                            | 15                | 0.001             |

----------------------

Results of Using InceptionV3 on the augmented-images-v3 dataset with various hyperparameters - 

With a learning rate of 0.001 the use of the Scheduler, we implemented a reduce learning rate on plateau where if the validation loss didn't improve after 3 epochs, the learning rate is then halved by a factor of 0.5. We implemented this scheduler for a variety of models and found that InceptionV3 performed the best, giving us an average validation accuracy of 71.17% and average training accuracy of 99%. Since this is a clear sign of overfitting, where the training accuracy far exceeds the validation accuracy, we turn to further fine tuning of the model.

We implemented early stopping into the InceptionV3 where if the validation loss doesn't improve after 5 epochs, early stopping would be triggered. In our case, the model scored an 60.89% for average validation accuracy and 91.51% for average training accuracy. The average validation accuracy is slightly worse than the reduce learning rate on plateau scheduler, thus requiring further fine tuning.

Modifying the final fully connected layer to include the following - 

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),  # Reduce features from 2048 to 512
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),  # Further reduce features to 256
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(dataset.classes))  # Final layer for 3 classes
    )

This was done to make the model less complex to adapt it to our dataset. However, with the early stopping implemented,
it yielded mediocre results, 62.52% average validation accuracy and 89.78% average training accuracy. Adding another Linear layer with 2048 inputs and 1024 outputs also didn't yield much improvement, giving us an average validation accuracy of 62.66% and 83.96% average training accuracy.

Initially, we assumed that the issue with the MRI images lied in the presence of the skull, eyes, and other irrelevant tissues when it comes to the classification of AD, CN, and MCI. Skull stripping involves the removal of non-brain tissue in MRI scans. Skull stripping has the advantage of removing tissue that isn't affected by Alzheimer's disease, which can potentially throw off the predictions of the pre-trained models, as they might focus more on the skull rather than the brain tissue, as suggested by [Impact of Skull Segmentation in MRI images for Alzheimer’s] paper 

A paper from Harvard student [citation required] detailed the implementation of SynthStrip, a tool that uses deep learning to remove the skeleton portion of the mri images, which has resulted in improvement in model performance (cite a paper). However, despite applying the skull stripping, InceptionV3 still performed poorly, with an average validation accuracy of 50.16% and average training accuracy of 72.80%. This was done via Early Stopping with a patient value of 8 epochs.

One potential issue discovered in the dataset is the fact that despite selecting the middle axial slice of each .nii file, the images tend to vary somewhat. This occured despite selecting the same MRI description for each .nii file, ensuring consistent preprocessing of each file. To cope this, standardization and normalization of the nifti files had to be performed, first starting with the orientation of the
nifti file to standard space via the FSL [citation] library. After this,  

