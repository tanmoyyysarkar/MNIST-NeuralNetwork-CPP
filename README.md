# MNIST-NeuralNetwork-CPP

# 🧠 Neural Network from Scratch (C++)

This project implements a simple feedforward neural network in **C++** trained on the **MNIST dataset**.
It is built entirely from scratch — no external machine learning libraries are used — to demonstrate how neural networks work under the hood.

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
.
├── headers/ # Header files (class definitions, function prototypes)
├── src/ # Source files (class implementations)
├── data/ # Dataset (MNIST files, ignored in git)
├── main.cpp # Entry point
├── utils.cpp/.hpp # Utility functions
├── Network.cpp/.hpp # Neural network implementation
├── README.md # Project documentation
└── .gitignore # Ignore datasets, build files, etc.

yaml
Copy
Edit

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Build the project
Using g++:

bash
Copy
Edit
g++ main.cpp src/*.cpp -o nn -O2
3. Run
bash
Copy
Edit
./nn
📊 Dataset
This project uses the MNIST handwritten digits dataset.
Download from Yann LeCun’s MNIST page and place the files inside the data/ folder.

Files needed:

train-images.idx3-ubyte

train-labels.idx1-ubyte

t10k-images.idx3-ubyte

t10k-labels.idx1-ubyte

⚠️ The dataset is not included in the repo (see .gitignore).

📖 How It Works
Feedforward – Computes activations layer by layer.

Backpropagation – Calculates gradients of cost function wrt weights & biases.

SGD – Updates weights in mini-batches using gradient descent.

Evaluation – Compares predictions with labels for accuracy.

✅ Results
On MNIST test set (10,000 digits), the network achieves around X% accuracy (after Y epochs).
(Fill this in once you run your model!)

🛠️ Future Improvements
Add ReLU activation

Support cross-entropy loss

Save & load trained models

Multi-threading for faster training

🤝 Contributing
Pull requests and suggestions are welcome!
If you’d like to contribute, fork the repo and open a PR.

📜 License
This project is licensed under the MIT License.
