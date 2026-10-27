# **ViT + PSO Implementation for Malaria Classification Task**

**Project Objective** 🔬

![ViT Illustration](assets/vit.jfif)

This project aims to build and optimize a deep learning model capable of classifying images of red blood cells as either **"Parasitized"** (infected with malaria) or **"Uninfected"**. The model employed is a **Vision Transformer (ViT)**, which is then further optimized using **Particle Swarm Optimization (PSO)** to find the best hyperparameter combination to achieve maximum classification performance.

**Methodology** ⚙️
1.  **Dataset**: This project uses a dataset of red blood cell images, divided into two classes. The data is then split into a training set (80%), a validation set (10%), and a test set (10%).
2.  **Model**: The base model used is a pre-trained Vision Transformer (`google/vit-base-patch16-224`) from Hugging Face, with its classifier head adapted for the binary classification task.
3.  **Preprocessing & Augmentation**: Images are processed to meet the input standards of the ViT model. Data augmentation techniques such as rotation, cropping, flipping, and color jittering are applied to the training data to improve model robustness.
4.  **Hyperparameter Optimization and Training**:
    * **Manual Testing**: An initial series of tests was conducted with several manual combinations of optimizers (Adam & AdamW), learning rates, and batch sizes to establish the best baseline model.
    * **PSO Optimization**: **Particle Swarm Optimization (PSO)** was implemented to automatically explore the hyperparameter search space and find the most optimal combination of learning rate and batch size, efficiently guiding the search towards the most promising configurations.
5.  **Evaluation**: The performance of the best model from manual testing and the PSO-optimized model are evaluated on the same unseen test set using metrics such as accuracy, precision, recall, f1-score, and a confusion matrix.

---

## **Comparative Results: Manual vs. PSO Optimization**

Here is a comparison of the results from manual testing versus the model whose hyperparameters were optimized by PSO.

### **1. Manual Hyperparameter Testing Results**

| Test | Optimizer | Learning Rate | Batch Size | Best Validation Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Adam** | **$10^{-4}$** | **32** | **97.64%** | **97.71%** |
| **2** | AdamW | $10^{-4}$ | 32 | 97.31% | 97.46% |
| **3** | AdamW | $10^{-3}$ | 64 | 96.08% | 96.12% |
| **4** | Adam | $10^{-3}$ | 64 | 96.08% | 95.75% |

### **2. PSO-Optimized Hyperparameter Results**

Based on the execution of the `pso-tuning.ipynb` file, the PSO algorithm found the following optimal hyperparameter configuration:

* **Optimizer**: Adam
* **Best Learning Rate found by PSO**: **$9.2 \times 10^{-5}$** (0.000092)
* **Best Batch Size found by PSO**: **32**
* **Final Test Accuracy (with PSO parameters)**: **97.97%**

---

## **Analysis and Conclusion**

1.  **Manual Search Findings**: From the manual tests, the best configuration was **Test 1** (Adam Optimizer, Learning Rate $10^{-4}$, Batch Size 32), which yielded a test accuracy of **97.71%**. This result indicated that a lower learning rate and a smaller batch size were more effective.

2.  **PSO Finds a Better Solution**: The **PSO algorithm successfully found a superior configuration** compared to the manual search results. With an even lower learning rate ($9.2 \times 10^{-5}$) and batch size (32), the model was able to achieve a higher test accuracy of **97.97%**.

3.  **The Power of Optimization**: The improvement from 97.71% to 97.97% demonstrates that automated hyperparameter search methods like PSO can explore the search space more granularly and find a "sweet spot" that is difficult to locate through manual testing. PSO proved that a more patient approach (very low learning rate) was highly effective for this task.

**Final Conclusion:**
The implementation of **Particle Swarm Optimization (PSO)** significantly improved the performance of the Vision Transformer model. The configuration found by PSO (Optimizer **Adam**, Learning Rate **$9.2 \times 10^{-5}$**, Batch Size **32**) proved to be the best, achieving an accuracy of **97.97%** on the test data. This proves that combining an advanced architecture like Vision Transformer with a metaheuristic optimization technique like PSO is a very powerful approach to achieving state-of-the-art results in medical image classification tasks.

**Model Source:** https://huggingface.co/docs/transformers/en/model_doc/vit#vision-transformer-vit

**Dataset Source:** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
