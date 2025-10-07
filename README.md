# CC_2023
Cognition and Computation(Deep Belief Networks on EMNIST dataset)

Author: Kateryna Skurativska

Professors: Alberto Testolin, Marco Zorzi

This project implements and analyzes Deep Belief Networks (DBNs) on the EMNIST dataset, aiming to explore representation learning, classification accuracy, and robustness to noise and adversarial attacks.

### **1. Goal of the project**

The project focuses on training a **Deep Belief Network (DBN)** to recognize handwritten letters from the EMNIST dataset and to compare its performance and robustness with a standard **feed-forward neural network (FFNN)**.
It also investigates how **unsupervised hierarchical feature learning** affects model interpretability and resistance to input perturbations (noise and adversarial attacks).


### **2. Dataset preparation**

* The **EMNIST “letters” split** was used, containing 26 letter classes plus one label index, with training and testing sets (124,800 and 20,800 samples respectively).
* Images were **normalized**, **rotated**, and converted into **tensors** suitable for PyTorch.


### **3. Model: Deep Belief Network (DBN)**

* The DBN architecture was built using pre-trained **Restricted Boltzmann Machines (RBMs)** stacked in three hidden layers of sizes `[500, 500, 1000]`.
* Each RBM learned unsupervised latent features through **Contrastive Divergence (CD-1)** training.
* The model was trained for **50 epochs** with a batch size of **125** using the `DBN.py` and `RBM.py` modules downloaded from an open-source PyTorch repository.


### **4. Visualization of learned features**

* After training, the project visualized **receptive fields** (the weight patterns learned by each hidden unit).
* Early-layer weights resembled **stroke-like features** (edges and curves of letters).
* Higher-layer weights represented more **abstract combinations** of letter components.

### **5. Clustering of learned representations**

* The internal feature representations from each DBN layer were extracted.
* The **mean hidden representation** for each letter class was computed.
* Using **hierarchical clustering**, the relationships among class centroids were analyzed, revealing that the DBN learned semantically meaningful clusters (e.g., letters with similar shapes grouped together).


### **6. Linear readout classification**

* Linear classifiers were trained on top of the hidden representations of each DBN layer to evaluate how discriminative each representation was.
* The **accuracy increased with depth**, showing that deeper layers captured more informative abstractions about the input data.

### **7. Comparison with a Feed-Forward Neural Network**

* A **feed-forward network** with the same architecture (784 → 500 → 500 → 1000 → 27) was trained end-to-end using supervised learning.
* The DBN with linear readouts was compared against this FFNN to assess the benefits of unsupervised pretraining.
* Both achieved good performance, but the DBN showed **greater robustness to input noise**.

### **8. Robustness to noise**

* Gaussian noise of increasing intensity was injected into the test images.
* Accuracy of both DBN and FFNN was measured as noise increased.
* DBN’s higher layers maintained **better stability and performance under noise**, suggesting stronger generalization and denoising capability.


### **9. Adversarial attacks**

* The **Fast Gradient Sign Method (FGSM)** was implemented to create **adversarial samples** that slightly modify inputs to fool the model.
* Both models (DBN and FFNN) were tested against varying attack strengths (ε).
* DBN showed **greater resilience** to adversarial perturbations, especially when using **one “top-down reconstruction” step** (reconstructing the input via the generative model before reclassification).
* Psychometric-like curves demonstrated how accuracy dropped with stronger attacks, with the DBN degrading more gracefully than the FFNN.

### **10. Key outcomes**

* The DBN successfully learned hierarchical representations of handwritten letters in an unsupervised manner.
* Higher-level features provided robust and semantically meaningful representations.
* Compared to a supervised FFNN:

  * The DBN was more **robust to noise and adversarial attacks**.
  * Its generative nature allowed **reconstruction-based defense mechanisms**.
* The project highlights the value of **unsupervised deep generative learning** for improving robustness and interpretability in neural networks.


### **In summary**

This project implemented a full pipeline for **training, visualizing, evaluating, and attacking a Deep Belief Network** on the EMNIST dataset. It compared unsupervised hierarchical learning with a traditional supervised feed-forward model and demonstrated the DBN’s **superior feature stability and robustness** under challenging input conditions.

