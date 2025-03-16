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

## Model Performance for augmented-images-v1 (horizontal flip and jitter)

| **Model**     | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** | **Epochs** | **Learing Rate** |
|---------------|----------------|--------------------|--------------|------------------|------------|------------------|
| ResNet50      | 0.9288         | 54.69%             | 0.9687       | 48.79%           | 10         |        NA        |
| VGGNET        | 0.0139         | 99.75%             | 1.5022       | 72.91%           | 20         |        NA        |
| AlexNet       | 0 .4233        | 81 .58%            | 0.8465       | 67.72%           | 20         |        NA        |
| InceptionV3   | 0.0134         | 99.72%             | 0.9121       | 73.52%           | 20         |        NA        |
| EffNetB3      | 0 .7023        | 67 .14%            | 0 .7755      | 62 .93%          | 20         |        NA        |
| EffNetB3      | 0.8974         | 55.86%             | 0.8681       | 56.72%           | 10         | 0.0001           |
| ResNet18      | 0.0214         | 99.59%             | 1.0446       | 71.79%           | 10         | 0.001            |

## Model Performance for augmented-images-v2 (horizontal flip and vertical flip)

| **Model**     | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** | **Epochs** | **Learing Rate** |
|---------------|----------------|--------------------|--------------|------------------|------------|------------------|
| ViT           | 0.0065         | 99.80%             | 2.8423       | 53.77%           | 25         | 0.001            |
| InceptionV3   | 0.1235         | 97.43%             | 0.9228       | 63.03%           | 15         | 0.0001           |

## Model Performance for augmented-images-v3 (horizontal flips, 900 ~ samples per class)

| Model         | Training Accuracy | Validation Accuracy | Epochs | **Learning Rate** |
|---------------|-------------------|---------------------|--------|-------------------|
| InceptionV3   | 99.28%            | 71.43%              | 15     | 0.001             |
| InceptionV3   | 92.18%            | 56.78%              | 15     | 0.0001            |
| EffcientNetB3 | 79.65%            | 59.86%              | 10     | 0.001             |
| AlextNet      | 51.24%            | 44.85%              | 15     | 0.001             |
| ViT           | 93.89%            | 46.47%              | 25     | 0.001             | 
