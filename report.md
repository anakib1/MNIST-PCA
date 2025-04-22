# MNIST Dataset Optimization using PCA

## Introduction

This report presents the results of an experiment aimed at optimizing the MNIST dataset using Principal Component Analysis (PCA). The goal was to identify and remove non-informative pixels while maintaining classification accuracy. This optimization can lead to more efficient neural network training and potentially better generalization by focusing on the most salient features.

## Methodology

### Dataset and Preprocessing
- Used the MNIST dataset (70,000 handwritten digits)
- Split into training (50,000) and validation (20,000) sets
- Normalized pixel values to [0, 1] range

### PCA Analysis
- Applied PCA to the training data to identify pixel importance based on variance explained.
- Tested 10 different mask sizes by keeping the top N% most important pixels (10% to 100%).
- For each mask size:
  - Calculated pixel importance by summing the absolute values of PCA components.
  - Created binary masks based on percentile thresholds of pixel importance.
  - Trained a neural network with the masked images.

### Neural Network Architecture
- Simple feedforward network with three layers:
  - Input layer (size varies with mask, from 79 to 748 pixels)
  - Hidden layer 1: 128 neurons with ReLU activation
  - Hidden layer 2: 64 neurons with ReLU activation
  - Output layer: 10 neurons (one per digit)
- Trained for 5 epochs using Adam optimizer (learning rate 0.001)
- Cross-entropy loss function

## Results

### Performance Analysis
The experiment tested different percentages of pixels kept, yielding the following validation accuracies (full details in `performance_results.txt`):

| % Pixels Kept | Number of Pixels | Validation Accuracy |
|---------------|-----------------|---------------------|
| 10%           | 79              | 92.98%              |
| 20%           | 157             | 95.60%              |
| 30%           | 235             | 96.69%              |
| 40%           | 314             | 97.11%              |
| 50%           | 392             | 97.14%              |
| 60%           | 470             | 97.29%              |
| 70%           | 549             | 97.38%              |
| 80%           | 627             | 97.27%              |
| 90%           | 705             | 97.31%              |
| 100%          | 748             | 97.42%              |

*(Note: 100% kept corresponds to 748 pixels, not 784, because pixels with zero variance across the training set were implicitly removed)*

### Key Findings

1.  **Pixel Importance Distribution:** The pixel importance heatmap (`masks/mask_Xpercent.png`) consistently shows that the central region of the 28x28 grid contains the most important pixels. Importance drops off significantly towards the edges, confirming that the borders contain less discriminative information for digit classification.

2.  **Performance vs Pixel Count:** Accuracy increases significantly as the percentage of kept pixels grows from 10% (92.98%) to 40% (97.11%). Beyond 40%, the accuracy gains are marginal, plateauing around 97.1-97.4%. Keeping 70% of pixels (549 pixels) achieved 97.38% accuracy, very close to the maximum accuracy of 97.42% obtained with 100% (748) pixels.

3.  **Optimal Mask Size Trade-off:** The results indicate a clear trade-off. Keeping around **40% to 70%** of the most important pixels (314 to 549 pixels) provides a good balance between model accuracy and computational efficiency. For instance, using only 40% of the pixels reduces the input dimensionality by 60% while sacrificing less than 0.4% absolute accuracy compared to using all informative pixels.

## Visual Analysis

The `masks` directory contains visualizations for each percentage tested.
- The **Pixel Mask** plots clearly show the mask growing from the center outwards as more pixels are kept.
- The **Pixel Importance** heatmap remains consistent across tests, highlighting the stable high importance of central pixels.
- The **Example Masked Image** plots demonstrate how applying the mask removes peripheral pixel information while preserving the core structure of the digit (in this case, the first training image). Even with only 50% of pixels, the essential shape of the digit remains visible.

These visualizations confirm that PCA successfully identifies spatially coherent and intuitively important regions for this dataset.

## Conclusions

1.  **Efficiency Gains:** PCA effectively identifies non-informative pixels in the MNIST dataset. Removing up to 60% of the pixels (keeping the top 40%) results in only a minor drop in accuracy (<0.4%) for the tested simple neural network. This significantly reduces the input dimensionality, leading to potential speedups in training and inference and lower memory usage.

2.  **Optimal Pixel Range:** For this specific setup, keeping between **40% and 70%** of the pixels offers the best compromise between performance and efficiency. The exact choice depends on the specific constraints and accuracy requirements of the application.

3.  **Generalizability:** The approach demonstrates the potential of using dimensionality reduction techniques like PCA for feature selection in image datasets, especially when dealing with images with consistent structures and non-informative backgrounds or borders.

## Files Generated

1.  `performance_analysis.png`: Plots visualizing accuracy vs. pixel count and accuracy vs. percentage kept.
2.  `performance_results.txt`: Detailed numerical results (pixel count, accuracy) for each percentage.
3.  `masks/`: Directory containing visualizations (`mask_Xpercent.png`) for each mask size, showing the mask, pixel importance, and an example masked digit.

## References

1.  MNIST Dataset: http://yann.lecun.com/exdb/mnist/
2.  Scikit-learn PCA Documentation
3.  PyTorch Neural Network Implementation 