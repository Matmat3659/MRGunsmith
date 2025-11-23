#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "json.hpp" // For loading the tags

using json = nlohmann::json;
using namespace std;

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ====================== UTILITIES AND ACTIVATIONS =====================

// ReLU (Rectified Linear Unit)
inline float relu(float x) { return std::max(0.0f, x); }

// Softplus (smooth approximation of ReLU, used here as a proxy for GELU)
inline float softplus(float x) { return std::log1p(std::exp(x)); }

// Sigmoid (logistic) - used mainly for output
inline float sigmoid(float x) {
    if (x < -20.0f) return 1e-7f;
    if (x > 20.0f) return 1.0f - 1e-7f;
    return 1.0f / (1.0f + std::exp(-x));
}

// Activation Enum
enum class Activation { RELU, SOFTPLUS, SIGMOID, IDENTITY };

// ---------------- Flatten ----------------
vector<float> flatten(const vector<vector<vector<float>>>& x){
    vector<float> out;
    for(const auto &f:x)
        for(const auto &row:f)
            for(float v:row)
                out.push_back(v);
    return out;
}

// ---------------- Image and Tag Loading ----------------

// Renamed from load_image_64 to be more general for any target size
vector<vector<vector<float>>> load_image_resized(string path, int targetSize){
    int w, h, c;
    // Load image as 3 channels (RGB)
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if(!img){ cerr << "Failed to load " << path << endl; exit(1); }

    // Output is [3][targetSize][targetSize]
    vector<vector<vector<float>>> out(3, vector<vector<float>>(targetSize, vector<float>(targetSize, 0)));
    const float inv255 = 1.0f / 255.0f;

    // Simple Bilinear Interpolation for Resizing
    for(int i=0; i<targetSize; i++){
        float fx = i * (h-1.0f)/(targetSize-1.0f); int x0 = (int)fx, x1 = min(x0+1, h-1); float dx = fx - x0;
        for(int j=0; j<targetSize; j++){
            float fy = j * (w-1.0f)/(targetSize-1.0f); int y0 = (int)fy, y1 = min(y0+1, w-1); float dy = fy - y0;

            for(int ch=0; ch<3; ch++){
                // Get 4 surrounding pixels (normalized 0-1)
                float v00 = img[(x0*w + y0)*3 + ch]*inv255;
                float v01 = img[(x0*w + y1)*3 + ch]*inv255;
                float v10 = img[(x1*w + y0)*3 + ch]*inv255;
                float v11 = img[(x1*w + y1)*3 + ch]*inv255;

                // Bilinear interpolation
                out[ch][i][j] = (1-dx)*(1-dy)*v00 + (1-dx)*dy*v01 + dx*(1-dy)*v10 + dx*dy*v11;
            }
        }
    }

    // Normalization to [-1, 1] range (as used in the original trainer)
    for(int ch=0; ch<3; ch++)
        for(int i=0; i<targetSize; i++)
            for(int j=0; j<targetSize; j++)
                out[ch][i][j] = (out[ch][i][j] - 0.5f) / 0.5f;

    stbi_image_free(img);
    return out;
}

void load_tags(map<int, string>& idx2tag, const string& modelPrefix){
    ifstream f(modelPrefix + "_tags.json");
    if(!f.is_open()){ cerr << "Cannot open tag file: " << modelPrefix << "_tags.json\n"; return; }
    json j; f >> j;
    idx2tag.clear();
    for(auto &item: j.items()){
        // Key is tag name (string), value is index (int)
        idx2tag[item.value()] = item.key();
    }
    f.close();
    cout << "Loaded " << idx2tag.size() << " class tags." << endl;
}

// ====================== LAYER IMPLEMENTATIONS (Inference Only) =====================

// ---------------- Layer Normalization ----------------
struct LayerNorm {
    int channels;
    std::vector<float> gamma; // Learnable scale
    std::vector<float> beta;  // Learnable shift

    LayerNorm(int C) : channels(C) {
        gamma.resize(C, 1.0f);
        beta.resize(C, 0.0f);
    }

    // Input shape: [C][H][W]
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        int H = input[0].size();
        int W = input[0][0].size();
        int N = H * W; // Elements per channel

        std::vector<std::vector<std::vector<float>>> output(channels, std::vector<std::vector<float>>(H, std::vector<float>(W)));
        const float epsilon = 1e-5f;

        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            float mean = 0.0f;
            // Calculate Mean
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    mean += input[c][i][j];
                }
            }
            mean /= N;

            // Calculate Variance
            float variance = 0.0f;
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float diff = input[c][i][j] - mean;
                    variance += diff * diff;
                }
            }
            variance /= N;

            float invStdDev = 1.0f / std::sqrt(variance + epsilon);

            // Normalize and apply scale/shift
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float normalized = (input[c][i][j] - mean) * invStdDev;
                    output[c][i][j] = gamma[c] * normalized + beta[c];
                }
            }
        }
        return output;
    }

    void load(const std::string& prefix){
        std::ifstream gfile(prefix + "_gamma.bin", std::ios::binary);
        if (gfile.is_open()) gfile.read(reinterpret_cast<char*>(gamma.data()), gamma.size()*sizeof(float));
        else throw std::runtime_error("Could not load: " + prefix + "_gamma.bin");
        gfile.close();

        std::ifstream bfile(prefix + "_beta.bin", std::ios::binary);
        if (bfile.is_open()) bfile.read(reinterpret_cast<char*>(beta.data()), beta.size()*sizeof(float));
        else throw std::runtime_error("Could not load: " + prefix + "_beta.bin");
        bfile.close();
    }
};

// ---------------- Convolution Layer (Inference Only) ----------------
struct ConvLayer {
    int kernelSize, numFilters, stride, padding;
    Activation act;
    int inChannels;
    int groups;

    std::vector<std::vector<std::vector<std::vector<float>>>> kernels;
    std::vector<float> biases;

    ConvLayer(int inChannels_, int kernelSize_, int numFilters_, Activation act_ = Activation::RELU,
              int stride_ = 1, int padding_ = 0, int groups_ = 1)
        : inChannels(inChannels_), kernelSize(kernelSize_), numFilters(numFilters_), act(act_),
          stride(stride_), padding(padding_), groups(groups_)
    {
        if (inChannels % groups != 0 || numFilters % groups != 0) {
            throw std::invalid_argument("InChannels and NumFilters must be divisible by groups.");
        }
        int channels_per_group = inChannels / groups;

        // Kernels shape: [numFilters][channels_per_group][kernelSize][kernelSize]
        kernels.resize(numFilters, std::vector<std::vector<std::vector<float>>>(
            channels_per_group,
            std::vector<std::vector<float>>(kernelSize, std::vector<float>(kernelSize))));
        biases.resize(numFilters, 0.0f);
    }

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        int n = input[0].size();
        int outSize = (n + 2*padding - kernelSize) / stride + 1;

        std::vector<std::vector<std::vector<float>>> out(numFilters, std::vector<std::vector<float>>(outSize, std::vector<float>(outSize, 0.0f)));

        int channels_per_group = inChannels / groups;
        int filters_per_group = numFilters / groups;

        #pragma omp parallel for collapse(3)
        for(int f=0; f<numFilters; f++){
            int group_idx = f / filters_per_group;

            for(int i=0; i<outSize; i++){
                for(int j=0; j<outSize; j++){
                    float sum = 0.0f;

                    for(int c_in_g=0; c_in_g<channels_per_group; c_in_g++){
                        int c_in = group_idx * channels_per_group + c_in_g;

                        for(int ki=0; ki<kernelSize; ki++)
                            for(int kj=0; kj<kernelSize; kj++){
                                int in_i = i*stride + ki - padding;
                                int in_j = j*stride + kj - padding;
                                float val = (in_i>=0 && in_i<n && in_j>=0 && in_j<n) ? input[c_in][in_i][in_j] : 0.0f;
                                sum += val * kernels[f][c_in_g][ki][kj];
                            }
                    }

                    sum += biases[f];

                    // Activation
                    switch(act){
                        case Activation::RELU: out[f][i][j] = relu(sum); break;
                        case Activation::SOFTPLUS: out[f][i][j] = softplus(sum); break;
                        case Activation::SIGMOID: out[f][i][j] = sigmoid(sum); break;
                        case Activation::IDENTITY: out[f][i][j] = sum; break;
                    }
                }
            }
        }

        return out;
    }

    void load(std::string prefix){
        std::ifstream kfile(prefix + "_kernels.bin", std::ios::binary);
        if (!kfile.is_open()) throw std::runtime_error("Could not load: " + prefix + "_kernels.bin");

        for(auto &f:kernels)
            for(auto &c:f)
                for(auto &row:c)
                    kfile.read(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        kfile.close();

        std::ifstream bfile(prefix + "_bias.bin", std::ios::binary);
        if (!bfile.is_open()) throw std::runtime_error("Could not load: " + prefix + "_bias.bin");
        bfile.read(reinterpret_cast<char*>(biases.data()), biases.size()*sizeof(float));
        bfile.close();
    }
};

// ---------------- ConvNeXt Block (Inference Only) ----------------
struct ConvNeXtBlock {
    int channels;
    ConvLayer dwConv;
    LayerNorm ln;
    ConvLayer pwConv1; // Expansion
    ConvLayer pwConv2; // Projection

    ConvNeXtBlock(int C, int expansion_ratio = 4)
        : channels(C),
        // 1. Depthwise Conv: 7x7, groups=C (Depthwise), padding=3 (to keep size).
        dwConv(C, 7, C, Activation::IDENTITY, 1, 3, C),
        // 2. LayerNorm: applied to the output of DW Conv
        ln(C),
        // 3. 1x1 Expansion: C in, 4*C out. SOFTPLUS (proxy for GELU) activation.
        pwConv1(C, 1, C * expansion_ratio, Activation::SOFTPLUS, 1, 0, 1),
        // 4. 1x1 Projection: 4*C in, C out. MUST be IDENTITY (no activation).
        pwConv2(C * expansion_ratio, 1, C, Activation::IDENTITY, 1, 0, 1)
    {}

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        // Residual Connection (Identity)
        std::vector<std::vector<std::vector<float>>> residual = input;

        // 1. Depthwise Conv 7x7
        std::vector<std::vector<std::vector<float>>> x = dwConv.forward(input);

        // 2. Layer Normalization
        x = ln.forward(x);

        // 3. 1x1 Pointwise Expansion (Softplus/GELU)
        x = pwConv1.forward(x);

        // 4. 1x1 Pointwise Projection (Linear/Identity)
        x = pwConv2.forward(x);

        // 5. Add Residual (x + residual)
        int C = x.size();
        int H = x[0].size();
        int W = x[0][0].size();

        #pragma omp parallel for collapse(3)
        for(int c=0; c<C; c++)
            for(int i=0; i<H; i++)
                for(int j=0; j<W; j++)
                    x[c][i][j] += residual[c][i][j];

        return x;
    }

    void load(const std::string& prefix) {
        dwConv.load(prefix + "_dw");
        ln.load(prefix + "_ln");
        pwConv1.load(prefix + "_pw1");
        pwConv2.load(prefix + "_pw2");
    }
};

// ---------------- Fully Connected Layer (Inference Only) ----------------
struct FCLayer {
    int inSize, outSize;
    std::vector<std::vector<float>> W;
    std::vector<float> B;
    Activation act;

    FCLayer(int inS, int outS, Activation act_ = Activation::SIGMOID)
        : inSize(inS), outSize(outS), act(act_)
    {
        W.resize(outSize, std::vector<float>(inSize));
        B.resize(outSize, 0.0f);
    }

    std::vector<float> forward(const std::vector<float>& in){
        std::vector<float> out(outSize);

        for(int i=0; i<outSize; i++){
            float s = 0;
            for(int j=0; j<inSize; j++)
                s += W[i][j] * in[j];
            s += B[i];

            // Activation
            switch(act){
                case Activation::RELU: out[i] = relu(s); break;
                case Activation::SOFTPLUS: out[i] = softplus(s); break;
                case Activation::SIGMOID: out[i] = sigmoid(s); break;
                case Activation::IDENTITY: out[i] = s; break;
            }
        }
        return out;
    }

    void load(const std::string& prefix){
        std::ifstream wfile(prefix + "_weights.bin", std::ios::binary);
        if (!wfile.is_open()) throw std::runtime_error("Could not load: " + prefix + "_weights.bin");

        for(auto &row: W)
            wfile.read(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        wfile.close();

        std::ifstream bfile(prefix + "_bias.bin", std::ios::binary);
        if (!bfile.is_open()) throw std::runtime_error("Could not load: " + prefix + "_bias.bin");
        bfile.read(reinterpret_cast<char*>(B.data()), B.size()*sizeof(float));
        bfile.close();
    }
};


// ====================== MAIN INFERENCE PROGRAM =====================
int main(int argc,char **argv){
    string modelPrefix = "";
    string imagePath = "";
    int imageSize = 128; // Default size assumed for the trained model

    // Command line argument parsing
    for(int i=1;i<argc;i++){
        string arg=argv[i];
        if(arg=="--model" && i+1<argc){ modelPrefix=argv[i+1]; i++; }
        else if(arg=="--image" && i+1<argc){ imagePath=argv[i+1]; i++; }
        else if(arg=="--size" && i+1<argc){ imageSize = stoi(argv[i+1]); i++; }
    }

    if(modelPrefix == "" || imagePath == ""){
        cerr << "Usage: " << argv[0] << " --model <model_prefix> --image <image_path> [--size <image_size>]\n";
        cerr << "Example model prefix: convnext_model\n";
        return 1;
    }

    try {
        // --- 1. Load Model Components ---
        const int C_base = 32;
        map<int, string> idx2tag;

        // Stage 0: Stem (k=4, s=4, 3 -> C_base) - Output (C_base, Size/4, Size/4)
        ConvLayer stem(3, 4, C_base, Activation::IDENTITY, 4, 0, 1);
        stem.load(modelPrefix + "_stem");

        // Stage 1: Block 1 (C=C_base) - Output (C_base, Size/4, Size/4)
        ConvNeXtBlock block1(C_base);
        block1.load(modelPrefix + "_block1");

        // Downsample 1: (k=2, s=2, C_base -> 2*C_base) - Output (2*C_base, Size/8, Size/8)
        ConvLayer downsample1(C_base, 2, C_base * 2, Activation::IDENTITY, 2, 0, 1);
        downsample1.load(modelPrefix + "_ds1");

        // Stage 2: Block 2 (C=2*C_base) - Output (2*C_base, Size/8, Size/8)
        ConvNeXtBlock block2(C_base * 2);
        block2.load(modelPrefix + "_block2");

        // Load tags to know output size
        load_tags(idx2tag, modelPrefix);
        if (idx2tag.empty()) {
            throw std::runtime_error("Failed to load class tags. Cannot initialize FC layer.");
        }
        int numClasses = idx2tag.size();
        int fcInSize = C_base * 2; // Output of Stage 2 is 2*C_base channels, which is the input to FC after Global Avg Pool

        // Classification Head: (2*C_base -> NumClasses)
        FCLayer fcLayer(fcInSize, numClasses, Activation::SIGMOID);
        fcLayer.load(modelPrefix + "_fc");
        
        cout << "Model components and weights loaded successfully.\n";

        // --- 2. Load and Preprocess Image ---
        vector<vector<vector<float>>> inputImage = load_image_resized(imagePath, imageSize);
        cout << "Image '" << imagePath << "' loaded and resized to " << imageSize << "x" << imageSize << ".\n";

        // --- 3. Forward Pass ---

        // Stage 0: Stem
        auto x = stem.forward(inputImage);

        // Stage 1: Block 1
        x = block1.forward(x);

        // Downsample 1
        x = downsample1.forward(x);

        // Stage 2: Block 2
        x = block2.forward(x);

        // Global Average Pooling (GAP)
        int C_out = x.size();
        int H_out = x[0].size();
        int W_out = x[0][0].size();
        int N_spatial = H_out * W_out;

        vector<float> pooledOutput(C_out, 0.0f);

        #pragma omp parallel for
        for (int c = 0; c < C_out; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    sum += x[c][i][j];
                }
            }
            pooledOutput[c] = sum / N_spatial;
        }

        // Classification Head
        vector<float> predictions = fcLayer.forward(pooledOutput);

        // --- 4. Display Results ---
        cout << "\n--- Prediction Results (Threshold 0.5) ---\n";
        for (int i = 0; i < numClasses; ++i) {
            float prob = predictions[i];
            cout << idx2tag[i] << ": " << fixed << setprecision(4) << prob;
            if (prob >= 0.5f) {
                cout << " [PREDICTED]";
            }
            cout << "\n";
        }

    } catch (const std::exception& e) {
        cerr << "\nFATAL ERROR during inference: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "\nFATAL UNKNOWN ERROR during inference.\n";
        return 1;
    }

    return 0;
}
