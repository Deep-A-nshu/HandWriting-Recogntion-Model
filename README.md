# HandWriting-Recogntion-Model
A Simple handwriting Recognition model build using CNN-RNN layers with CTC loss and decode functions trained on the IAM lines dataset.
The model divided the data into batches of size 10. Each batch is trained using **7 layers of Convolution Neural Network(CNN)** and **2 layers of
Bidirectional-Recurrent Neural Network(RNN)** which used **RMS Propagation Optimizer** at a **learning rate of 0.001**. The **Connectionist Temporal Classification(CTC) Loss function** is used and the **CTC decode function** to decode the output of the neural network. CTC loss function is used because there is no direct mapping between the input image and the ground truth text. 

**This model is currently in initail training-testing phase.**

### Requirements
1. Python3 (version <=3.7.x)
2. Tensorflow 1.15.5 (or any tensorflow 1.x version)
   `pip install tensorflow==1.15.5`
3. Numpy
   `pip install numpy`
4. OpenCV2
   `pip install opencv-python`
5. Autocorrect 
   `pip install autocorrect`

### How to run
* **Download the IAM-lines dataset**
  * You can download the IAM **lines** dataset from [Here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)
  * Place this downloaded dataset in the **data folder**.
* **Run the code**
  * Run the main.py file from the src folder directory
  `python main.py`

**Citations to be added**

