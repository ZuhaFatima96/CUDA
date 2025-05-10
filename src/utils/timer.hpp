#pragma once

#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace utils {

// Timer class for performance measurements
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Initialize matrix with random values
template<typename T>
void initializeMatrix(std::vector<T>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-10.0, 10.0);
    
    for (int i = 0; i < size; ++i) {
        matrix[i] = dis(gen);
    }
}

// Compare results between CPU and GPU implementations
template<typename T>
bool compareResults(const std::vector<T>& cpu_result, const std::vector<T>& gpu_result, 
                   T tolerance = 1e-6) {
    if (cpu_result.size() != gpu_result.size()) {
        std::cout << "Size mismatch!" << std::endl;
        return false;
    }

    T total_diff = 0;
    int mismatches = 0;

    for (size_t i = 0; i < cpu_result.size(); ++i) {
        T diff = std::abs(cpu_result[i] - gpu_result[i]);
        total_diff += diff;
        if (diff > tolerance) {
            mismatches++;
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] 
                         << " GPU=" << gpu_result[i] << " (diff=" << diff << ")" << std::endl;
            }
        }
    }

    std::cout << "Total difference: " << total_diff << std::endl;
    std::cout << "Number of mismatches: " << mismatches << std::endl;

    return mismatches == 0;
}

// Save performance results to CSV
void savePerformanceResults(const std::string& filename, 
                          const std::vector<std::pair<std::string, double>>& results) {
    std::ofstream outFile(filename);
    outFile << "Implementation,Time(ms)\n";
    
    for (const auto& result : results) {
        outFile << result.first << "," << result.second << "\n";
    }
}

// Generate performance plot using Python (requires matplotlib)
void generatePlot(const std::string& data_file, const std::string& output_file) {
    std::string cmd = "python3 -c '\
import matplotlib.pyplot as plt\n\
import pandas as pd\n\
import seaborn as sns\n\
data = pd.read_csv(\"" + data_file + "\")\n\
plt.figure(figsize=(10, 6))\n\
sns.barplot(x=\"Implementation\", y=\"Time(ms)\", data=data)\n\
plt.xticks(rotation=45)\n\
plt.tight_layout()\n\
plt.savefig(\"" + output_file + "\")\n'";
    
    system(cmd.c_str());
}

} // namespace utils
