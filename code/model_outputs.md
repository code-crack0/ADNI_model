# Model Performance Summary

## **Inference Accuracy**
| Model          | Accuracy |
|---------------|----------|
| **AlexNet**   | 17.81%   |
| **ResNet18**  | 52.89%   |
| **EfficientNet** | 17.78% |
| **VGG16**     | 17.46%   |

---

### Model Performance with Unbalanced Dataset

| Model          | Train Accuracy | Validation Accuracy| Test Accuracy |
|---------------|----------|----------|------------|
| **AlexNet**   | 52.89%   | 52.75% | 52.99% |
| **ResNet18**  | 62.17%   | 43.04% | 42.16% |
| **EfficientNetb3** | 83.28% | 59.87% | 58.80% |  
| **VGG16**     |  52.89%  | 52.75% | 52.99% | 
| **InceptionV3** |  51.78%  | 47.25% | 47.98% | 

---

## **Training Details (ResNet50)**
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
