#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

#include "Matrix.h"
#include "DataLoader.h"

// ---------------------------------------------------------
// Neural Network Training
// ---------------------------------------------------------
int main() {
    // --- Dataset Configuration ---
    string train_file = "train.csv";
    string test_file = "test.csv";
    int label_col = 0;
    double divide_by = 255.0; // Use 255.0 for MNIST, 1.0 for standard datasets
    
    // --- Hyperparameters ---
    int hidden_dim = 128;
    double lr = 0.1;
    int epochs = 20;
    int batch_size = 128;

    cout << "Loading Datasets..." << endl;
    
    Matrix X_train, X_test;
    vector<int> y_train;
    vector<int> dummy_y;
    
    load_csv(train_file, X_train, y_train, label_col, divide_by);
    load_csv(test_file, X_test, dummy_y, -1, divide_by);

    if (X_train.r == 0 || X_test.r == 0) {
        cerr << "Could not load data. Ensure files are valid." << endl;
        return 1;
    }

    int input_dim = X_train.c;
    int m = X_train.r;

    // Auto-detect number of classes (output_dim) based on highest label value
    int output_dim = 1; 
    for(int i = 0; i < m; i++) {
        if (y_train[i] >= output_dim) {
            output_dim = y_train[i] + 1;
        }
    }

    cout << "Training Samples: " << m << ", Features: " << input_dim << ", Classes: " << output_dim << endl;

    // Kaiming Initialization
    mt19937 gen(42);
    normal_distribution<double> d1(0.0, sqrt(2.0 / input_dim));
    normal_distribution<double> d2(0.0, sqrt(2.0 / hidden_dim));

    Matrix W1(input_dim, hidden_dim);
    Matrix b1(1, hidden_dim, 0.0);
    Matrix W2(hidden_dim, output_dim);
    Matrix b2(1, output_dim, 0.0);

    for (size_t i = 0; i < W1.d.size(); ++i) W1.d[i] = d1(gen);
    for (size_t i = 0; i < W2.d.size(); ++i) W2.d[i] = d2(gen);

    cout << "Training 1-Hidden-Layer Neural Network (Mini-Batch SGD)..." << endl;
    
    int num_batches = m / batch_size;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        // Shuffle indices for SGD
        vector<int> indices(m);
        for(int i=0; i<m; ++i) indices[i] = i;
        shuffle(indices.begin(), indices.end(), gen);

        for (int b = 0; b < num_batches; ++b) {
            int start = b * batch_size;
            int end = min(start + batch_size, m);
            int current_batch_size = end - start;

            // Extract mini-batch
            Matrix X_batch(current_batch_size, input_dim);
            vector<int> y_batch(current_batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start + i];
                y_batch[i] = y_train[idx];
                for (int j = 0; j < input_dim; ++j) {
                    X_batch(i, j) = X_train(idx, j);
                }
            }

            // --- Forward Pass ---
            Matrix Z1 = dot(X_batch, W1);
            for (int i = 0; i < Z1.r; ++i) 
                for (int j = 0; j < Z1.c; ++j) Z1(i, j) += b1(0, j);

            Matrix A1(Z1.r, Z1.c); // ReLU
            for (size_t i = 0; i < Z1.d.size(); ++i) A1.d[i] = max(0.0, Z1.d[i]);

            Matrix Z2 = dot(A1, W2);
            for (int i = 0; i < Z2.r; ++i) 
                for (int j = 0; j < Z2.c; ++j) Z2(i, j) += b2(0, j);

            Matrix A2(Z2.r, Z2.c); // Softmax
            for (int i = 0; i < Z2.r; ++i) {
                double max_z = Z2(i, 0);
                for (int j = 1; j < output_dim; ++j) max_z = max(max_z, Z2(i, j));
                
                double sum_exp = 0.0;
                for (int j = 0; j < output_dim; ++j) {
                    A2(i, j) = exp(Z2(i, j) - max_z);
                    sum_exp += A2(i, j);
                }
                
                int pred_class = 0;
                double best_p = -1;
                for (int j = 0; j < output_dim; ++j) {
                    A2(i, j) /= sum_exp;
                    if (A2(i, j) > best_p) {
                        best_p = A2(i, j);
                        pred_class = j;
                    }
                }
                if (pred_class == y_batch[i]) epoch_correct++;
                epoch_loss -= log(max(A2(i, y_batch[i]), 1e-15));
            }

            // --- Backward Pass ---
            Matrix dZ2 = A2;
            for (int i = 0; i < current_batch_size; ++i) dZ2(i, y_batch[i]) -= 1.0;

            Matrix A1_T = transpose(A1);
            Matrix dW2 = dot(A1_T, dZ2);
            
            Matrix W2_T = transpose(W2);
            Matrix dA1 = dot(dZ2, W2_T);
            
            Matrix dZ1(dA1.r, dA1.c);
            for (size_t i = 0; i < dA1.d.size(); ++i) {
                dZ1.d[i] = (Z1.d[i] > 0) ? dA1.d[i] : 0.0; // ReLU derivative
            }

            Matrix X_T = transpose(X_batch);
            Matrix dW1 = dot(X_T, dZ1);

            // --- Update Weights ---
            for (size_t i = 0; i < W2.d.size(); ++i) W2.d[i] -= (lr / current_batch_size) * dW2.d[i];
            for (size_t i = 0; i < W1.d.size(); ++i) W1.d[i] -= (lr / current_batch_size) * dW1.d[i];

            for (int i = 0; i < current_batch_size; ++i) {
                for (int j = 0; j < output_dim; ++j) b2(0, j) -= (lr / current_batch_size) * dZ2(i, j);
                for (int j = 0; j < hidden_dim; ++j) b1(0, j) -= (lr / current_batch_size) * dZ1(i, j);
            }
        }

        if (epoch % 10 == 0 || epoch == 1) {
            cout << "Epoch " << epoch << "/" << epochs 
                 << " | Loss: " << (epoch_loss / m) 
                 << " | Accuracy: " << ((double)epoch_correct / m) * 100.0 << "%" << endl;
        }
    }

    // ---------------------------------------------------------
    // Inference on test.csv
    // ---------------------------------------------------------
    cout << "\nRunning Inference on MNIST test set..." << endl;
    
    Matrix Z1 = dot(X_test, W1);
    for (int i = 0; i < Z1.r; ++i) 
        for (int j = 0; j < Z1.c; ++j) Z1(i, j) += b1(0, j);

    Matrix A1(Z1.r, Z1.c);
    for (size_t i = 0; i < Z1.d.size(); ++i) A1.d[i] = max(0.0, Z1.d[i]);

    Matrix Z2 = dot(A1, W2);
    for (int i = 0; i < Z2.r; ++i) 
        for (int j = 0; j < Z2.c; ++j) Z2(i, j) += b2(0, j);

    ofstream out("submission.csv");
    out << "ImageId,Label\n";

    for (int i = 0; i < Z2.r; ++i) {
        int best_class = 0;
        double max_z = Z2(i, 0);
        for (int j = 1; j < output_dim; ++j) {
            if (Z2(i, j) > max_z) {
                max_z = Z2(i, j);
                best_class = j;
            }
        }
        out << (i + 1) << "," << best_class << "\n"; // ImageId is 1-indexed
    }

    out.close();
    cout << "Predictions saved to submission.csv!" << endl;

    return 0;
}
