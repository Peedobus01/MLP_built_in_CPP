#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

void load_csv(const string& filename, Matrix& X, vector<int>& y, int label_col, double divide_by) {
    ifstream file(filename);
    string line, val;
    if (!file.is_open()) {
        cerr << "Failed to open " << filename << endl;
        return;
    }
    
    getline(file, line); // Skip header
    
    vector<vector<double>> temp_x;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<double> row;
        
        int current_col = 0;
        while (getline(ss, val, ',')) {
            if (current_col == label_col) {
                y.push_back(stoi(val)); 
            } else {
                row.push_back(stod(val) / divide_by); 
            }
            current_col++;
        }
        temp_x.push_back(row);
    }
    
    if (temp_x.empty()) return;
    X = Matrix(temp_x.size(), temp_x[0].size());
    for (size_t i = 0; i < temp_x.size(); ++i) {
        for (size_t j = 0; j < temp_x[0].size(); ++j) {
            X(i, j) = temp_x[i][j];
        }
    }
}
