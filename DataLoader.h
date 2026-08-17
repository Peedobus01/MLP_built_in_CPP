#ifndef DATALOADER_H
#define DATALOADER_H

#include "Matrix.h"
#include <string>
#include <vector>

// load_csv takes a label_col parameter and a divide_by parameter for normalization.
// If label_col is -1, it means there are no labels in this dataset.
// Otherwise, it represents the 0-indexed column number of the label.
// divide_by defaults to 1.0 (no division), but for images you might pass 255.0
void load_csv(const std::string& filename, Matrix& X, std::vector<int>& y, int label_col = -1, double divide_by = 1.0);

#endif // DATALOADER_H
