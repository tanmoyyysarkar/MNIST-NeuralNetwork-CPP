# 🧠 Neural Network from Scratch (C++)

A mini-project which implements a simple feedforward neural network in **C++** trained on the **MNIST dataset**.
It is built entirely from scratch (no external machine learning libraries are used) to demonstrate how neural networks work under the hood.

---

## ✨ Features
- Implementation of feedforward neural network
- Stochastic Gradient Descent (SGD) with mini-batches
- Backpropagation algorithm
- Sigmoid activation function
- Cost function (Quadratic)
- Trained & tested on the MNIST dataset

---

## 📂 Project Structure
```
MNIST-NeuralNetwork-CPP/
├── headers/
│   ├── Network.hpp
│   └── Utils.hpp
├── .gitignore
├── main.cpp
├── Network.cpp
├── utils.cpp
└── README.md
```
---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/tanmoyyysarkar/MNIST-NeuralNetwork-CPP
cd MINST-NeuralNetwork-CPP
```
### 2. Download the Dataset

This project uses the MNIST handwritten digits dataset.
Make a new directory named *Data*
```bash
mkdir Data
```
Download the data set from [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place the files inside the Data/ folder.

Files needed:

- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

⚠️ The dataset is not included in the repo (see .gitignore).

### 3. Build the project
Using g++:
```bash
g++ -O2 main.cpp Network.cpp utils.cpp -o main
```
### 4. Run
```bash
./main
```
## 📖 How It Works
- Feedforward – Computes activations layer by layer.
- Backpropagation – Calculates gradients of cost function wrt weights & biases.
- SGD – Updates weights in mini-batches using gradient descent.
- Evaluation – Compares predictions with labels for accuracy.

## ✅ Results
On MNIST test set (10,000 digits), the network achieves around 72% accuracy (after 50 epochs).

## 🛠️ Future Improvements
- Add ReLU activation
- Support cross-entropy loss
- Save & load trained models
- Multi-threading for faster training

## 🙌 Credits
- MNIST dataset by [Yann LeCun, Corinna Cortes, Christopher J.C. Burges](http://yann.lecun.com/exdb/mnist/)
- Inspired by Michael Nielsen’s book: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- Special thanks to the **C++ community** and open-source contributors for reference implementations and discussions.

## 🤝 Contributing
Pull requests and suggestions are welcome!
If you’d like to contribute, fork the repo and open a PR.

## 📜 License
This project is licensed under the MIT License.
