# Multilayer Perceptron (MLP) built in C++

This project implements a fully connected feedforward Neural Network (Multilayer Perceptron) entirely from scratch in C++, without using any external machine learning or linear algebra libraries.

It was originally built to solve the Kaggle MNIST Digit Recognizer challenge but has been generalized to support any custom CSV dataset.

## Features

- **Built from scratch:** Implements the Forward Pass, Backward Pass (Backpropagation), and Mini-Batch Stochastic Gradient Descent (SGD) manually.
- **Modular Design:**
  - `Matrix.cpp` / `Matrix.h`: Custom 2D Matrix structure with fast `dot()` product and `transpose()` operations.
  - `DataLoader.cpp` / `DataLoader.h`: Flexible CSV parser that automatically handles column splitting, label extraction, and normalization.
  - `main.cpp`: The core Neural Network training loop and architecture definition.
- **Generalized:** Automatically detects the number of output classes based on your training data.

## Requirements

- A standard C++ compiler (like GCC / `g++`).
- Data files named `train.csv` and `test.csv` placed in the root folder. *(Note: The Kaggle MNIST dataset files are large and are generally not uploaded to GitHub).*

## How to Compile and Run

Because this project is split into multiple files, you must compile them all together. Using the `-O3` flag is highly recommended as it aggressively optimizes the math loops for maximum speed.

**1. Open your terminal in the project directory.**

**2. Compile the code:**
```bash
g++ -O3 main.cpp Matrix.cpp DataLoader.cpp -o main.exe
```

**3. Run the compiled executable:**
```bash
.\main.exe
```

## Using Custom Datasets

To use a dataset other than MNIST, open `main.cpp` and update the configuration section at the top of the `main()` function:
```cpp
// --- Dataset Configuration ---
string train_file = "your_train_file.csv";
string test_file = "your_test_file.csv";
int label_col = 0; // The column index (0-based) containing the answers
double divide_by = 1.0; // Use 255.0 for image pixels, 1.0 for standard numeric datasets
```
