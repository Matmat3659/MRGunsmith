#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <omp.h>
#include <tuple> // For std::tie in BCE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- Activation ----------------

// ReLU (Rectified Linear Unit)
inline float relu(float x) { return std::max(0.0f, x); }
inline float relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// Softplus (smooth approximation of ReLU, used here as a proxy for GELU)
inline float softplus(float x) { return std::log1p(std::exp(x)); }
inline float softplus_deriv(float x) {
    // derivative of softplus is sigmoid
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}

// Sigmoid (logistic) - used mainly for BCE output and Softplus derivative
inline float sigmoid(float x) {
    // Handle overflow prevention
    if (x < -20.0f) return 1e-7f;
    if (x > 20.0f) return 1.0f - 1e-7f;
    return 1.0f / (1.0f + std::exp(-x));
}
inline float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// ====================== ACT ENUM =====================
enum class Activation { RELU, SOFTPLUS, SIGMOID, IDENTITY };

// ---------------- Xavier Initialization ----------------
float xavier_init(int fan_in, int fan_out) {
    // Simple He/Kaiming initialization approximation
    thread_local std::mt19937 gen(std::random_device{}());
    // For ReLU-like activations (SOFTPLUS is close enough)
    float limit = std::sqrt(2.0f / fan_in);
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

// ---------------- Flatten / Unflatten ----------------
vector<float> flatten(const vector<vector<vector<float>>>& x){
    vector<float> out;
    for(const auto &f:x)
        for(const auto &row:f)
            for(float v:row)
                out.push_back(v);
    return out;
}

vector<vector<vector<float>>> unflatten(const vector<float>& flat, int F, int H, int W){
    if(flat.size() != size_t(F*H*W)){
        throw std::runtime_error("unflatten: flat vector size does not match target shape");
    }

    vector<vector<vector<float>>> out(F, vector<vector<float>>(H, vector<float>(W)));
    int k = 0;
    for(int f=0; f<F; f++)
        for(int i=0; i<H; i++)
            for(int j=0; j<W; j++)
                out[f][i][j] = flat[k++];
    return out;
}

// ---------------- Layer Normalization ----------------
struct LayerNorm {
    int channels;
    std::vector<float> gamma; // Learnable scale
    std::vector<float> beta;  // Learnable shift

    // Cached values for backward pass
    std::vector<float> lastInputFlat; // [C * H * W]
    std::vector<float> lastMean;      // [C]
    std::vector<float> lastInvStdDev; // [C]

    LayerNorm(int C) : channels(C) {
        gamma.resize(C, 1.0f); // Initialize scale to 1
        beta.resize(C, 0.0f);  // Initialize shift to 0
    }

    // Input shape: [C][H][W]
    // LayerNorm normalizes across the spatial dimensions (H, W) for each channel (C) independently.
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        int H = input[0].size();
        int W = input[0][0].size();
        int N = H * W; // Elements per channel

        // Flatten the input and resize caches
        lastInputFlat.assign(channels * N, 0.0f);
        lastMean.assign(channels, 0.0f);
        lastInvStdDev.assign(channels, 0.0f);

        std::vector<std::vector<std::vector<float>>> output(channels, std::vector<std::vector<float>>(H, std::vector<float>(W)));
        const float epsilon = 1e-5f;

        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            float mean = 0.0f;
            // Calculate Mean
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float val = input[c][i][j];
                    mean += val;
                    lastInputFlat[c * N + i * W + j] = val;
                }
            }
            mean /= N;
            lastMean[c] = mean;

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
            lastInvStdDev[c] = invStdDev;

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

    // dOut shape: [C][H][W]
    // Returns dInput [C][H][W]
    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& dOut, float lr) {
        int H = dOut[0].size();
        int W = dOut[0][0].size();
        int N = H * W;
        const float epsilon = 1e-5f;

        std::vector<std::vector<std::vector<float>>> dInput(channels, std::vector<std::vector<float>>(H, std::vector<float>(W)));

        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            float mean = lastMean[c];
            float invStdDev = lastInvStdDev[c];

            // 1. Calculate dGamma and dBeta
            float dGamma = 0.0f;
            float dBeta = 0.0f;

            // dNormalized = dOut * gamma
            std::vector<float> dNormalized(N);

            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    int flatIdx = c * N + i * W + j;
                    float x_minus_mu = lastInputFlat[flatIdx] - mean;
                    float normalized = x_minus_mu * invStdDev;

                    float grad_out = dOut[c][i][j];
                    dGamma += grad_out * normalized;
                    dBeta += grad_out;

                    dNormalized[i * W + j] = grad_out * gamma[c];
                }
            }

            // Update gamma and beta
            gamma[c] -= lr * dGamma;
            beta[c] -= lr * dBeta;

            // 2. Calculate dInput (dL/dX)
            float term1_sum = 0.0f;
            float term2_sum = 0.0f;

            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    int flatIdx = c * N + i * W + j;
                    float x_minus_mu = lastInputFlat[flatIdx] - mean;
                    float dN = dNormalized[i * W + j];

                    term1_sum += dN * x_minus_mu;
                    term2_sum += dN;
                }
            }

            float invN = 1.0f / N;
            float invN_stddev_pow3 = invN * invStdDev * invStdDev * invStdDev;

            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    int flatIdx = c * N + i * W + j;
                    float x_minus_mu = lastInputFlat[flatIdx] - mean;
                    float dN = dNormalized[i * W + j];

                    float dX_i = dN * invStdDev;
                    dX_i -= x_minus_mu * term1_sum * invN_stddev_pow3;
                    dX_i -= term2_sum * invStdDev * invN;

                    dInput[c][i][j] = dX_i;
                }
            }
        }
        return dInput;
    }

    void save(const std::string& prefix){
        std::ofstream gfile(prefix + "_gamma.bin", std::ios::binary);
        gfile.write(reinterpret_cast<char*>(gamma.data()), gamma.size()*sizeof(float));
        gfile.close();

        std::ofstream bfile(prefix + "_beta.bin", std::ios::binary);
        bfile.write(reinterpret_cast<char*>(beta.data()), beta.size()*sizeof(float));
        bfile.close();
    }

    void load(const std::string& prefix){
        std::ifstream gfile(prefix + "_gamma.bin", std::ios::binary);
        gfile.read(reinterpret_cast<char*>(gamma.data()), gamma.size()*sizeof(float));
        gfile.close();

        std::ifstream bfile(prefix + "_beta.bin", std::ios::binary);
        bfile.read(reinterpret_cast<char*>(beta.data()), beta.size()*sizeof(float));
        bfile.close();
    }
};

// ---------------- Convolution Layer (Modified to support Depthwise) ----------------
struct ConvLayer {
    int kernelSize, numFilters, stride, padding;
    Activation act;
    int inChannels;
    int groups; // groups=inChannels means Depthwise Conv

    std::vector<std::vector<std::vector<std::vector<float>>>> kernels;
    std::vector<float> biases;

    std::vector<std::vector<std::vector<float>>> lastInput;
    std::vector<std::vector<std::vector<float>>> lastPreActivation;

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

        // Initialize weights
        #pragma omp parallel for collapse(4)
        for(int f=0; f<numFilters; f++)
            for(int c=0; c<channels_per_group; c++)
                for(int i=0; i<kernelSize; i++)
                    for(int j=0; j<kernelSize; j++)
                        kernels[f][c][i][j] = xavier_init(kernelSize*kernelSize*channels_per_group, numFilters/groups);
    }

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        lastInput = input;
        int n = input[0].size();
        int outSize = (n + 2*padding - kernelSize) / stride + 1;

        lastPreActivation.assign(numFilters, std::vector<std::vector<float>>(outSize, std::vector<float>(outSize, 0.0f)));
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
                    lastPreActivation[f][i][j] = sum;

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

    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& dOut, float lr){
        int n = lastInput[0].size();
        int outSize = dOut[0].size();
        std::vector<std::vector<std::vector<float>>> dInput(inChannels, std::vector<std::vector<float>>(n, std::vector<float>(n,0.0f)));

        int channels_per_group = inChannels / groups;
        int filters_per_group = numFilters / groups;

        #pragma omp parallel for
        for(int f=0; f<numFilters; f++){
            int group_idx = f / filters_per_group;

            for(int i=0; i<outSize; i++){
                for(int j=0; j<outSize; j++){
                    float grad = 0.0f;
                    float z = lastPreActivation[f][i][j];

                    // Apply derivative of activation
                    switch(act){
                        case Activation::RELU: grad = dOut[f][i][j]*relu_deriv(z); break;
                        case Activation::SOFTPLUS: grad = dOut[f][i][j]*softplus_deriv(z); break;
                        case Activation::SIGMOID: grad = dOut[f][i][j]*sigmoid_deriv(z); break;
                        case Activation::IDENTITY: grad = dOut[f][i][j]; break; // Derivative of f(x)=x is 1
                    }

                    biases[f] -= lr*grad;

                    for(int c_in_g=0; c_in_g<channels_per_group; c_in_g++){
                        int c_in = group_idx * channels_per_group + c_in_g;

                        for(int ki=0; ki<kernelSize; ki++)
                            for(int kj=0; kj<kernelSize; kj++){
                                int in_i = i*stride + ki - padding;
                                int in_j = j*stride + kj - padding;
                                if(in_i>=0 && in_i<n && in_j>=0 && in_j<n){
                                    // Gradient w.r.t input (dInput)
                                    #pragma omp atomic
                                    dInput[c_in][in_i][in_j] += grad * kernels[f][c_in_g][ki][kj];

                                    // Gradient w.r.t weight (kernel update)
                                    kernels[f][c_in_g][ki][kj] -= lr*grad*lastInput[c_in][in_i][in_j];
                                }
                            }
                    }
                }
            }
        }

        return dInput;
    }

    void save(std::string prefix){
        std::ofstream kfile(prefix + "_kernels.bin", std::ios::binary);
        for(auto &f:kernels)
            for(auto &c:f)
                for(auto &row:c)
                    kfile.write(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        kfile.close();

        std::ofstream bfile(prefix + "_bias.bin", std::ios::binary);
        bfile.write(reinterpret_cast<char*>(biases.data()), biases.size()*sizeof(float));
        bfile.close();
    }

    void load(std::string prefix){
        std::ifstream kfile(prefix + "_kernels.bin", std::ios::binary);
        for(auto &f:kernels)
            for(auto &c:f)
                for(auto &row:c)
                    kfile.read(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        kfile.close();

        std::ifstream bfile(prefix + "_bias.bin", std::ios::binary);
        bfile.read(reinterpret_cast<char*>(biases.data()), biases.size()*sizeof(float));
        bfile.close();
    }
};

// ---------------- ConvNeXt Block ----------------
// Represents the core ConvNeXt inverted bottleneck block
struct ConvNeXtBlock {
    int channels;
    // 1. Depthwise Conv 7x7 (large kernel)
    ConvLayer dwConv;
    // 2. Layer Norm
    LayerNorm ln;
    // 3. 1x1 Pointwise Expansion (ratio 4, Softplus/GELU)
    ConvLayer pwConv1;
    // 4. 1x1 Pointwise Projection (back to C channels, Identity)
    ConvLayer pwConv2;

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

    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& dOut, float lr) {

        std::vector<std::vector<std::vector<float>>> dResidual = dOut;
        std::vector<std::vector<std::vector<float>>> dBlock = dOut;

        // 1. Backprop 1x1 Projection (pwConv2)
        dBlock = pwConv2.backward(dBlock, lr);

        // 2. Backprop 1x1 Expansion (pwConv1)
        dBlock = pwConv1.backward(dBlock, lr);

        // 3. Backprop Layer Norm
        dBlock = ln.backward(dBlock, lr);

        // 4. Backprop Depthwise Conv (dwConv)
        dBlock = dwConv.backward(dBlock, lr);

        // 5. Combine with residual gradient (dBlock + dResidual)
        int C = dBlock.size();
        int H = dBlock[0].size();
        int W = dBlock[0][0].size();

        #pragma omp parallel for collapse(3)
        for(int c=0; c<C; c++)
            for(int i=0; i<H; i++)
                for(int j=0; j<W; j++)
                    dBlock[c][i][j] += dResidual[c][i][j];

        return dBlock;
    }

    void save(const std::string& prefix) {
        dwConv.save(prefix + "_dw");
        ln.save(prefix + "_ln");
        pwConv1.save(prefix + "_pw1");
        pwConv2.save(prefix + "_pw2");
    }

    void load(const std::string& prefix) {
        dwConv.load(prefix + "_dw");
        ln.load(prefix + "_ln");
        pwConv1.load(prefix + "_pw1");
        pwConv2.load(prefix + "_pw2");
    }
};

// ---------------- MaxPool Layer (kept for downsampling) ----------------
struct MaxPool {
    int poolSize;
    int stride;
    int padding;
    // Stores the (i, j) coordinates of the maximum value for each output cell
    std::vector<std::vector<std::vector<std::pair<int,int>>>> maxPos;

    MaxPool(int poolSize_ = 2, int stride_ = 2, int padding_ = 0)
        : poolSize(poolSize_), stride(stride_), padding(padding_) {}

    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input)
    {
        int F = input.size();
        int H = input[0].size();
        int W = input[0][0].size();
        int outH = (H + 2*padding - poolSize) / stride + 1;
        int outW = (W + 2*padding - poolSize) / stride + 1;

        maxPos.resize(F, std::vector<std::vector<std::pair<int,int>>>(outH, std::vector<std::pair<int,int>>(outW)));

        std::vector<std::vector<std::vector<float>>> out(
            F, std::vector<std::vector<float>>(outH, std::vector<float>(outW, 0)));

        #pragma omp parallel for
        for(int f = 0; f < F; f++){
            for(int i = 0; i < outH; i++){
                for(int j = 0; j < outW; j++){
                    float m = -1e30f; // Very small number
                    int max_i = -1, max_j = -1;
                    for(int pi = 0; pi < poolSize; pi++){
                        for(int pj = 0; pj < poolSize; pj++){
                            int in_i = i*stride + pi - padding;
                            int in_j = j*stride + pj - padding;
                            float val = (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) ? input[f][in_i][in_j] : -1e30f;
                            if(val > m){ m = val; max_i = in_i; max_j = in_j; }
                        }
                    }
                    out[f][i][j] = m;
                    maxPos[f][i][j] = {max_i, max_j};
                }
            }
        }

        return out;
    }

    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& dOut, int origH, int origW)
    {
        int F = dOut.size();
        int outH = dOut[0].size();
        int outW = dOut[0][0].size();

        std::vector<std::vector<std::vector<float>>> dInput(
            F, std::vector<std::vector<float>>(origH, std::vector<float>(origW, 0)));

        #pragma omp parallel for
        for(int f = 0; f < F; f++){
            for(int i = 0; i < outH; i++){
                for(int j = 0; j < outW; j++){
                    // Use std::tie to extract the coordinates
                    int pi, pj;
                    std::tie(pi, pj) = maxPos[f][i][j];
                    if(pi >= 0 && pj >= 0)
                        dInput[f][pi][pj] = dOut[f][i][j];
                }
            }
        }

        return dInput;
    }
};

// ---------------- Fully Connected Layer (kept for classification head) ----------------
struct FCLayer {
    int inSize, outSize;
    std::vector<std::vector<float>> W;
    std::vector<float> B;
    std::vector<float> lastIn;
    std::vector<float> lastPreActivation; // To store z = Wx + b
    Activation act;
    float dropoutProb;
    std::vector<bool> dropoutMask;

    FCLayer(int inS, int outS, Activation act_ = Activation::SIGMOID, float dropout_=0.0f)
        : inSize(inS), outSize(outS), act(act_), dropoutProb(dropout_)
    {
        W.resize(outSize, std::vector<float>(inSize));
        B.resize(outSize, 0.0f);
        for(int i=0; i<outSize; i++)
            for(int j=0; j<inSize; j++)
                W[i][j] = xavier_init(inS, outS);
    }

    std::vector<float> forward(const std::vector<float>& in, bool training=true){
        lastIn = in;
        lastPreActivation.resize(outSize);
        std::vector<float> out(outSize);
        dropoutMask.clear();
        dropoutMask.resize(outSize, true);

        for(int i=0; i<outSize; i++){
            float s = 0;
            for(int j=0; j<inSize; j++)
                s += W[i][j] * in[j];
            s += B[i];
            lastPreActivation[i] = s;

            // Activation
            switch(act){
                case Activation::RELU: out[i] = relu(s); break;
                case Activation::SOFTPLUS: out[i] = softplus(s); break;
                case Activation::SIGMOID: out[i] = sigmoid(s); break;
                case Activation::IDENTITY: out[i] = s; break;
            }

            // Apply dropout if training
            if(training && dropoutProb > 0.0f){
                // Using rand() for simplicity. Scale is 1/(1-p).
                float scale = 1.0f / (1.0f - dropoutProb);
                float r = ((float)rand() / (float)RAND_MAX);
                if(r < dropoutProb){
                    out[i] = 0;
                    dropoutMask[i] = false;
                } else {
                    out[i] *= scale; // Apply scaling to compensate for dropped units
                }
            }
        }
        return out;
    }

    std::vector<float> backward(const std::vector<float>& grad, float lr){
        std::vector<float> dInput(inSize,0);

        // Calculate dL/dW and dL/dB (updates) and dL/dX (dInput)
        for(int i=0; i<outSize; i++){
            float g = grad[i];

            // Reapply dropout mask and scaling from forward pass
            if(!dropoutMask[i]) {
                g = 0.0f;
            } else if (dropoutProb > 0.0f) {
                float scale = 1.0f / (1.0f - dropoutProb);
                g *= scale;
            }


            // Apply derivative of activation if NOT using BCE_grad with SIGMOID
            float z = lastPreActivation[i];

            switch(act){
                case Activation::RELU: g *= relu_deriv(z); break;
                case Activation::SOFTPLUS: g *= softplus_deriv(z); break;
                case Activation::IDENTITY: g *= 1.0f; break;
                // BCE_grad already calculates dL/dz = p-t, so no further activation derivative is needed for SIGMOID.
                case Activation::SIGMOID: break;
            }

            for(int j=0; j<inSize; j++){
                dInput[j] += g * W[i][j];
                W[i][j] -= lr * g * lastIn[j]; // Update weights
            }
            B[i] -= lr * g; // Update bias
        }

        return dInput;
    }

    void save(const std::string& prefix){
        std::ofstream wfile(prefix + "_weights.bin", std::ios::binary);
        for(auto &row: W)
            wfile.write(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        wfile.close();

        std::ofstream bfile(prefix + "_bias.bin", std::ios::binary);
        bfile.write(reinterpret_cast<char*>(B.data()), B.size()*sizeof(float));
        bfile.close();
    }

    void load(const std::string& prefix){
        std::ifstream wfile(prefix + "_weights.bin", std::ios::binary);
        for(auto &row: W)
            wfile.read(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float));
        wfile.close();

        std::ifstream bfile(prefix + "_bias.bin", std::ios::binary);
        bfile.read(reinterpret_cast<char*>(B.data()), B.size()*sizeof(float));
        bfile.close();
    }
};

// ---------------- Helper Functions ----------------
vector<float> BCE_grad(const vector<float>& p, const vector<float>& t){
    vector<float> g(p.size());
    for(size_t i=0;i<p.size();i++){
        // p is the output of sigmoid (post-activation)
        float pi = std::clamp(p[i], 1e-7f, 1.0f-1e-7f);
        // This is dL/dz where z is the pre-sigmoid output (Wz+b)
        // dL/dz = (p - t)
        g[i] = pi - t[i];
    }
    return g;
}

void save_tags(const map<string,int>& tag2idx, const string& prefix){
    json j;
    for(auto &p: tag2idx) j[p.first] = p.second;
    ofstream f(prefix + "_tags.json");
    f << j.dump(4);
    f.close();
}

void load_tags(map<string,int>& tag2idx, const string& prefix){
    ifstream f(prefix + "_tags.json");
    if(!f.is_open()){ cerr << "Cannot open tag file\n"; return; }
    json j; f >> j;
    tag2idx.clear();
    for(auto &item: j.items()){
        tag2idx[item.key()] = item.value();
    }
    f.close();
}

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
        // Map target pixel i to source float coordinate fx (scaled from h-1)
        float fx = i * (h-1.0f)/(targetSize-1.0f); int x0 = (int)fx, x1 = min(x0+1, h-1); float dx = fx - x0;
        for(int j=0; j<targetSize; j++){
            // Map target pixel j to source float coordinate fy (scaled from w-1)
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

    // Normalization to [-1, 1] range
    for(int ch=0; ch<3; ch++)
        for(int i=0; i<targetSize; i++)
            for(int j=0; j<targetSize; j++)
                out[ch][i][j] = (out[ch][i][j] - 0.5f) / 0.5f;

    stbi_image_free(img);
    return out;
}

void load_dataset(const string& folder,
                  vector<vector<vector<vector<float>>>>& X,
                  vector<vector<float>>& Y,
                  map<string,int>& tag2idx,
                  int targetSize)
{
    ifstream tfile(folder + "/tags.json");
    if(!tfile.is_open()){ cerr<<"Cannot open tags.json\n"; exit(1);}
    json j; tfile >> j;

    int numTags = 0;
    for(auto &f:j.items())
        for(auto &img:f.value().items())
            for(auto &tag:img.value())
                if(tag2idx.find(tag) == tag2idx.end())
                    tag2idx[tag] = numTags++;

    for(auto &f:j.items())
        for(auto &img:f.value().items()){
            string path = folder + "/" + f.key() + "/" + img.key();
            // Use the renamed/updated image loading function
            auto imgRGB = load_image_resized(path, targetSize);
            X.push_back(imgRGB);

            vector<float> multi(tag2idx.size(), 0.0f);
            for(auto &tag: img.value())
                multi[tag2idx[tag]] = 1.0f;
            Y.push_back(multi);
        }
}

float multilabel_accuracy(const vector<float>& pred, const vector<float>& target, float threshold=0.5f){
    int correct=0, total=0;
    for(size_t i=0; i<pred.size(); i++){
        bool p = pred[i] > threshold;
        bool t = target[i] > 0.5f;
        if(p==t) correct++;
        total++;
    }
    return correct/(float)total;
}

// ---------------- MAIN ----------------
int main(int argc,char **argv){
    // Use current time for seeding random number generator
    srand(time(0));
    random_device rd; mt19937 g(rd());

    vector<vector<vector<vector<float>>>> X; // Dataset images
    vector<vector<float>> Y; // Dataset labels
    map<string,int> tag2idx; // Tag name to index mapping

    string dataFolder="Data";
    int epochs=50;
    float lr=0.005f;
    int batchSize=2;
    int numThreads = 4;
    // --- UPGRADE: Increased default image size from 64 to 128 ---
    int imageSize = 128;

    // Command line argument parsing
    for(int i=1;i<argc;i++){
        string arg=argv[i];
        if(arg=="--data" && i+1<argc){ dataFolder=argv[i+1]; i++; }
        else if(arg=="--epochs" && i+1<argc){ epochs=stoi(argv[i+1]); i++; }
        else if(arg=="--lr" && i+1<argc){ lr=stof(argv[i+1]); i++; }
        else if(arg=="--batch" && i+1<argc){ batchSize=stoi(argv[i+1]); i++; }
        else if(arg=="--threads" && i+1<argc){ numThreads = stoi(argv[i+1]); i++; }
        else if(arg=="--size" && i+1<argc){ imageSize = stoi(argv[i+1]); i++; } // New option to override size
    }

    omp_set_num_threads(numThreads);
    cout << "Using " << numThreads << " OpenMP threads\n";

    cout<<"Loading data from folder: "<<dataFolder<<" (Target size: "<<imageSize<<"x"<<imageSize<<")\n";
    load_dataset(dataFolder, X, Y, tag2idx, imageSize);
    cout<<"Loaded "<<X.size()<<" images, "<<tag2idx.size()<<" tags.\n";

    if (X.empty()) {
        cerr << "Error: No data loaded. Exiting.\n";
        return 1;
    }

    // --- ConvNeXt Tiny-like Architecture (Simplified) ---
    // The architecture is now more robust for 128x128 input.

    // Stage 1: Stem and first block (3 -> C1 channels)
    const int C1 = 24; // Initial channel width
    // Stem: 4x4 Conv, stride 4. Reduces 128x128 -> 32x32
    ConvLayer stem_conv(3, 4, C1, Activation::IDENTITY, 4, 0, 1);
    LayerNorm stem_ln(C1);
    ConvNeXtBlock block1(C1);

    // Stage 2: Downsampling and second block (C1 -> C2 channels)
    const int C2 = 48; // Second channel width
    // Downsample: MaxPool 2x2, stride 2. Reduces 32x32 -> 16x16
    MaxPool downsample2(2, 2);
    ConvLayer expand2(C1, 1, C2, Activation::IDENTITY, 1, 0, 1); // Expand channels C1 -> C2
    ConvNeXtBlock block2(C2);

    // Classifier Head
    int fcInputSize = 0;
    // Dry run forward pass to calculate the input size for the FC layer
    {
        auto s = stem_conv.forward(X[0]);
        s = stem_ln.forward(s);
        s = block1.forward(s);
        s = downsample2.forward(s);
        s = expand2.forward(s);
        s = block2.forward(s);
        fcInputSize = s.size() * s[0].size() * s[0][0].size();
    }
    // Final layer is SIGMOID for multi-label classification
    FCLayer classifier(fcInputSize, tag2idx.size(), Activation::SIGMOID);

    cout << "Network initialized.\n";
    cout << "Output feature map size (before flattening): " << C2 << "x" << sqrt(fcInputSize / C2) << "x" << sqrt(fcInputSize / C2) << endl;
    cout << "Input size to FC: " << fcInputSize << " (C="<< C2 <<")\n";

    // ---------------- Training loop ----------------
    for(int e=0; e<epochs; e++){
        float totalLoss = 0;

        // Shuffle the data for better stochasticity
        vector<int> indices(X.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), g);

        for(size_t i=0; i<X.size(); i += batchSize){
            // Mini-batch processing
            vector<vector<float>> batch_preds;
            vector<vector<float>> batch_targets;
            vector<vector<vector<vector<float>>>> batch_inputs;
            vector<float> avg_loss_grads(tag2idx.size(), 0.0f);

            int currentBatchSize = 0;
            for (int b = 0; b < batchSize && i + b < X.size(); ++b) {
                int idx = indices[i + b];
                batch_inputs.push_back(X[idx]);
                batch_targets.push_back(Y[idx]);
                currentBatchSize++;
            }

            // --- Forward Pass (Batch size 1 for now, or process sequentially) ---
            vector<vector<float>> all_dInput_fc; // Gradients from FC to Conv
            for (int b = 0; b < currentBatchSize; ++b) {
                // Fwd
                auto s = stem_conv.forward(batch_inputs[b]);
                s = stem_ln.forward(s);
                s = block1.forward(s);
                int h1 = s[0].size(); int w1 = s[0][0].size();

                s = downsample2.forward(s);
                s = expand2.forward(s);
                s = block2.forward(s);
                int h2 = s[0].size(); int w2 = s[0][0].size();

                auto flat = flatten(s);
                auto pred = classifier.forward(flat);

                batch_preds.push_back(pred);

                // Calculate Loss and Gradient w.r.t classifier output (dL/dz)
                auto grad_fc = BCE_grad(pred, batch_targets[b]);
                
                // Binary Cross Entropy Loss calculation
                for(size_t k=0; k<pred.size(); k++){
                    float p_k = std::clamp(pred[k], 1e-7f, 1.0f-1e-7f);
                    if(batch_targets[b][k] > 0.5f){
                        totalLoss += -std::log(p_k);
                    } else {
                        totalLoss += -std::log(1.0f - p_k);
                    }
                }

                // Accumulate gradients for the backward pass
                all_dInput_fc.push_back(grad_fc);
            }

            // --- Backward Pass ---
            // Process the batch in reverse, applying gradient averaging implicitly through the learning rate scale
            for (int b = 0; b < currentBatchSize; ++b) {
                // Backprop FC
                auto dFlat = classifier.backward(all_dInput_fc[b], lr / currentBatchSize); // Divide LR by batch size for averaging

                // Unflatten and Backprop Conv Blocks
                auto dConv = unflatten(dFlat, C2, block2.dwConv.lastInput[0].size(), block2.dwConv.lastInput[0][0].size());

                dConv = block2.backward(dConv, lr / currentBatchSize);
                dConv = expand2.backward(dConv, lr / currentBatchSize);
                dConv = downsample2.backward(dConv, block1.dwConv.lastInput[0].size(), block1.dwConv.lastInput[0][0].size());

                dConv = block1.backward(dConv, lr / currentBatchSize);
                dConv = stem_ln.backward(dConv, lr / currentBatchSize);
                dConv = stem_conv.backward(dConv, lr / currentBatchSize);
            }
        }

        // --- Epoch End Metrics ---
        float avgLoss = totalLoss / X.size();
        float avgAcc = 0;
        for(size_t i=0; i<X.size(); i++){
            // Run a quick inference (no backprop/dropout)
            auto s = stem_conv.forward(X[i]);
            s = stem_ln.forward(s);
            s = block1.forward(s);
            s = downsample2.forward(s);
            s = expand2.forward(s);
            s = block2.forward(s);
            auto pred = classifier.forward(flatten(s), false);
            avgAcc += multilabel_accuracy(pred, Y[i]);
        }
        avgAcc /= X.size();

        cout << "Epoch " << e+1 << "/" << epochs << ": Loss = " << avgLoss << ", Accuracy = " << avgAcc << endl;
    }

    // --- Save Model ---
    string modelName = "convnext_model";
    stem_conv.save(modelName + "_stem_conv");
    stem_ln.save(modelName + "_stem_ln");
    block1.save(modelName + "_block1");
    expand2.save(modelName + "_expand2");
    block2.save(modelName + "_block2");
    classifier.save(modelName + "_fc");
    save_tags(tag2idx, modelName);

    cout << "Model training complete and saved to " << modelName << "_*.bin/json\n";

    return 0;
}
