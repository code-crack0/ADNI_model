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

## **Training Details (Augmentation)**

### ResNet50
| **Epoch** | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** |
|-----------|----------------|--------------------|--------------|------------------|
| 1         | 1.0394         | 48.42%             | 0.9872       | 53.31%           |
| 2         | 0.9890         | 52.06%             | 1.1293       | 49.27%           |
| 3         | 0.9812         | 52.71%             | 1.1625       | 32.15%           |
| 4         | 0.9790         | 52.55%             | 1.0030       | 52.83%           |
| 5         | 0.9765         | 51.82%             | 0.9682       | 52.83%           |
| 6         | 0.9513         | 52.67%             | 1.0748       | 55.74%           |
| 7         | 0.9344         | 53.88%             | 0.9762       | 54.77%           |
| 8         | 0.9209         | 53.96%             | 0.9279       | 53.63%           |
| 9         | 0.8993         | 54.28%             | 1.0337       | 52.83%           |
| 10        | 0.9288         | 54.69%             | 0.9687       | 48.79%           |

### VGGNET
| **Epoch** | **Train Loss** | **Train Acc** | **Val Loss** | **Val Acc** |
|-----------|----------------|---------------|--------------|-------------|
| 1 | 0.9410 | 52.98% | 0.8962 | 55.40% |
| 2 | 0.8910 | 55.45% | 0.8714 | 54.58% |
| 3 | 0.8725 | 56.75% | 0.9189 | 52.34% |
| 4 | 0.8404 | 59.20% | 0.8180 | 60.39% |
| 5 | 0.8008 | 61.77% | 0.8276 | 59.16% |
| 6 | 0.7624 | 63.45% | 0.7817 | 63.75% |
| 7 | 0.7191 | 65.72% | 0.8170 | 59.06% |
| 8 | 0.6475 | 69.92% | 0.7638 | 63.95% |
| 9 | 0.6116 | 71.63% | 0.8337 | 63.54% |
| 10 | 0.5639 | 74.91% | 0.7409 | 67.01% |
| 11 | 0.5085 | 77.00% | 0.7395 | 67.31% |
| 12 | 0.4720 | 79.57% | 0.9960 | 63.85% |
| 13 | 0.4072 | 82.27% | 0.8588 | 65.58% |
| 14 | 0.3849 | 83.39% | 0.8150 | 66.29% |
| 15 | 0.2976 | 88.33% | 0.8209 | 66.90% |
| 16 | 0.1346 | 94.50% | 1.1635 | 72.30% |
| 17 | 0.0821 | 97.22% | 1.4679 | 67.72% |
| 18 | 0.0520 | 98.32% | 1.6459 | 70.47% |
| 19 | 0.0584 | 97.94% | 1.4108 | 71.59% |
| 20 | 0.0139 | 99.75% | 1.5022 | 72.91% |
