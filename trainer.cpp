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
#include <numeric>
#include <omp.h>
#include <tuple>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Assuming these two headers are available in the build environment
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "json.hpp" // json.hpp is still used for the dataset's tags file

using json = nlohmann::json;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Type Definitions for Eigen ---
using Tensor3f = Eigen::Tensor<float, 3, Eigen::RowMajor>; // Channel, Height, Width
using Tensor4f = Eigen::Tensor<float, 4, Eigen::RowMajor>; // Out, In, K, K
using Vectorf = Eigen::VectorXf;
using Matrixf = Eigen::MatrixXf;

// --- Global Optimization Parameters ---
constexpr float BETA1 = 0.9f;
constexpr float BETA2 = 0.999f;
constexpr float EPSILON = 1e-8f;
constexpr float WEIGHT_DECAY = 0.05f; // L2 regularization (ConvNeXt standard)

// ---------------- Binary I/O Utilities ----------------

template<typename T>
void write_bin(ofstream& ofs, const T* data, size_t size) {
    if (size > 0) {
        ofs.write(reinterpret_cast<const char*>(data), size * sizeof(T));
    }
}

template<typename T>
void read_bin(ifstream& ifs, T* data, size_t size) {
    if (size > 0) {
        ifs.read(reinterpret_cast<char*>(data), size * sizeof(T));
    }
}

// ---------------- Activation Functions ----------------

// Sigmoid and its derivative
inline float sigmoid(float x) {
    if (x < -20.0f) return 1e-7f;
    if (x > 20.0f) return 1.0f - 1e-7f;
    return 1.0f / (1.0f + std::exp(-x));
}

// Gaussian Error Linear Unit (GELU) Approximation (Standard for ConvNeXt)
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

inline float gelu_deriv(float x) {
    // Simplified derivative for the approximation (fast and effective)
    float tanh_arg = std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float sech2 = 1.0f - std::tanh(tanh_arg) * std::tanh(tanh_arg);
    float deriv_tanh_arg = std::sqrt(2.0f / M_PI) * (1.0f + 3.0f * 0.044715f * x * x);
    
    return 0.5f * (1.0f + std::tanh(tanh_arg)) + 0.5f * x * sech2 * deriv_tanh_arg;
}

// ReLU
inline float relu(float x) { return std::max(0.0f, x); }
inline float relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

enum class Activation { GELU, RELU, SIGMOID, IDENTITY };

// ---------------- Initialization and Schedule ----------------

float xavier_init(int fan_in, int fan_out) {
    thread_local std::mt19937 gen(std::random_device{}());
    float limit = std::sqrt(2.0f / fan_in);
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

// Cosine Annealing LR Schedule with Warmup
float cosine_annealing_lr(float base_lr, int current_epoch, int total_epochs, float warmup_epochs) {
    if (current_epoch < warmup_epochs) {
        // Linear Warmup
        return base_lr * (static_cast<float>(current_epoch) / warmup_epochs);
    }
    // Cosine Annealing
    float t = static_cast<float>(current_epoch - warmup_epochs);
    float T = static_cast<float>(total_epochs - warmup_epochs);
    if (T <= 0) return base_lr;
    return base_lr * 0.5f * (1.0f + std::cos(M_PI * t / T));
}

// ---------------- Global Average Pooling ----------------
struct GlobalAvgPool {
    GlobalAvgPool() {}

    Vectorf forward(const Tensor3f& input) {
        int C = input.dimension(0);
        int H = input.dimension(1);
        int W = input.dimension(2);
        int N = H * W;
        Vectorf out(C);

        #pragma omp parallel for
        for (int c = 0; c < C; ++c) {
            Eigen::Map<const Matrixf> mat(input.data() + c * N, H, W);
            out[c] = mat.mean();
        }
        return out;
    }

    Tensor3f backward(const Vectorf& dOut, int H, int W) {
        int C = dOut.size();
        Tensor3f dInput(C, H, W);
        float invN = 1.0f / (H * W);

        #pragma omp parallel for collapse(3)
        for (int c = 0; c < C; ++c) {
            float grad_c = dOut[c] * invN;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dInput(c, h, w) = grad_c;
                }
            }
        }
        return dInput;
    }
    // No parameters to save/load
};

// ---------------- Layer Normalization ----------------
struct LayerNorm {
    int channels;
    Vectorf gamma; // Learnable scale
    Vectorf beta;  // Learnable shift
    
    // Adam moments
    Vectorf M_gamma, V_gamma;
    Vectorf M_beta, V_beta;

    // Backward pass state
    Tensor3f lastInput;
    Vectorf lastMean;
    Vectorf lastInvStdDev;

    LayerNorm(int C) : channels(C) {
        gamma.resize(C); gamma.setOnes();
        beta.resize(C); beta.setZero();
        M_gamma.resize(C); M_gamma.setZero();
        V_gamma.resize(C); V_gamma.setZero();
        M_beta.resize(C); M_beta.setZero();
        V_beta.resize(C); V_beta.setZero();
    }

    Tensor3f forward(const Tensor3f& input) {
        int H = input.dimension(1);
        int W = input.dimension(2);
        int N = H * W; // H*W elements per channel

        lastInput = input;
        lastMean.resize(channels);
        lastInvStdDev.resize(channels);

        Tensor3f output(channels, H, W);
        const float epsilon = 1e-5f;

        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            Eigen::Map<const Matrixf> mat(input.data() + c * N, H, W);
            float mean = mat.mean();
            
            Matrixf diff = mat.array() - mean;
            float variance = diff.array().pow(2).mean();

            float invStdDev = 1.0f / std::sqrt(variance + epsilon);

            lastMean[c] = mean;
            lastInvStdDev[c] = invStdDev;

            Matrixf normalized = diff.array() * invStdDev;
            
            Eigen::Map<Matrixf> out_mat(output.data() + c * N, H, W);
            out_mat = (normalized.array() * gamma[c] + beta[c]).matrix();
        }
        return output;
    }

    Tensor3f backward(const Tensor3f& dOut, float lr, int t) {
        int H = dOut.dimension(1);
        int W = dOut.dimension(2);
        int N = H * W;
        const float invN = 1.0f / N;

        Tensor3f dInput(channels, H, W);

        // Adam learning rate with bias correction
        float lr_t = lr * std::sqrt(1.0f - std::pow(BETA2, t)) / (1.0f - std::pow(BETA1, t));

        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            float mean = lastMean[c];
            float invStdDev = lastInvStdDev[c];

            Eigen::Map<const Matrixf> dOut_mat(dOut.data() + c * N, H, W);
            Eigen::Map<const Matrixf> input_mat(lastInput.data() + c * N, H, W);

            Matrixf x_minus_mu = input_mat.array() - mean;
            // Matrixf normalized = x_minus_mu.array() * invStdDev; // Recompute normalized or use saved? Saved state is more memory efficient.

            // Gradients for gamma and beta
            Matrixf normalized_from_input = x_minus_mu.array() * invStdDev; // Recompute for accuracy
            float dGamma = (dOut_mat.array() * normalized_from_input.array()).sum();
            float dBeta = dOut_mat.sum();

            // Adam Update for gamma (no decay)
            M_gamma[c] = BETA1 * M_gamma[c] + (1.0f - BETA1) * dGamma;
            V_gamma[c] = BETA2 * V_gamma[c] + (1.0f - BETA2) * dGamma * dGamma;
            gamma[c] -= lr_t * M_gamma[c] / (std::sqrt(V_gamma[c]) + EPSILON);

            // Adam Update for beta (no decay)
            M_beta[c] = BETA1 * M_beta[c] + (1.0f - BETA1) * dBeta;
            V_beta[c] = BETA2 * V_beta[c] + (1.0f - BETA2) * dBeta * dBeta;
            beta[c] -= lr_t * M_beta[c] / (std::sqrt(V_beta[c]) + EPSILON);

            // Gradient for normalized input
            Matrixf dNormalized = dOut_mat.array() * gamma[c];

            // Calculate gradient for the input (dInput)
            float term1_sum = (dNormalized.array() * x_minus_mu.array()).sum();
            float term2_sum = dNormalized.sum();

            float invN_stddev_pow3 = invN * invStdDev * invStdDev * invStdDev;

            Matrixf dX_i = dNormalized.array() * invStdDev;
            dX_i.array() -= x_minus_mu.array() * term1_sum * invN_stddev_pow3;
            dX_i.array() -= term2_sum * invStdDev * invN;
            
            Eigen::Map<Matrixf> dInput_mat(dInput.data() + c * N, H, W);
            dInput_mat = dX_i;
        }
        return dInput;
    }

    void save(const std::string& prefix) {
        ofstream ofs(prefix + ".bin", ios::binary);
        if (!ofs.is_open()) { cerr << "Cannot open " << prefix << ".bin for LayerNorm saving\n"; return; }

        int C_size = channels;
        write_bin(ofs, &C_size, 1);
        write_bin(ofs, gamma.data(), gamma.size());
        write_bin(ofs, beta.data(), beta.size());
    }
    
    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { cerr << "Cannot open " << prefix << ".bin for LayerNorm loading\n"; return; }

        int C_size;
        read_bin(ifs, &C_size, 1);
        if (C_size != channels) {
            cerr << "Channel size mismatch in LayerNorm " << prefix << endl;
            return;
        }

        read_bin(ifs, gamma.data(), gamma.size());
        read_bin(ifs, beta.data(), beta.size());
    }
};

// ---------------- Convolution Layer ----------------
struct ConvLayer {
    int kernelSize, numFilters, stride, padding;
    Activation act;
    int inChannels;
    int groups;

    Tensor4f kernels;
    Vectorf biases;
    
    // Adam moments
    Tensor4f M_kernels, V_kernels;
    Vectorf M_biases, V_biases;

    // Backward pass state
    Tensor3f lastInput;
    Tensor3f lastPreActivation;

    ConvLayer(int inChannels_, int kernelSize_, int numFilters_, Activation act_ = Activation::RELU,
              int stride_ = 1, int padding_ = 0, int groups_ = 1)
        : inChannels(inChannels_), kernelSize(kernelSize_), numFilters(numFilters_), act(act_),
          stride(stride_), padding(padding_), groups(groups_)
    {
        if (inChannels % groups != 0 || numFilters % groups != 0) {
            throw std::invalid_argument("InChannels and NumFilters must be divisible by groups.");
        }
        int channels_per_group = inChannels / groups;

        kernels.resize(numFilters, channels_per_group, kernelSize, kernelSize);
        biases.resize(numFilters); biases.setZero();
        M_kernels.resize(kernels.dimensions()); M_kernels.setZero();
        V_kernels.resize(kernels.dimensions()); V_kernels.setZero();
        M_biases.resize(numFilters); M_biases.setZero();
        V_biases.resize(numFilters); V_biases.setZero();

        float limit = std::sqrt(2.0f / (kernelSize*kernelSize*channels_per_group));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::mt19937 gen(std::random_device{}());

        for(int i=0; i<kernels.size(); ++i) {
            kernels.data()[i] = dist(gen);
        }
    }

    Tensor3f forward(const Tensor3f& input) {
        lastInput = input;
        
        Eigen::DSizes<Eigen::Index, 3> input_dims = input.dimensions();
        int H_in = input_dims[1];
        int W_in = input_dims[2];
        
        int H_out = (H_in + 2 * padding - kernelSize) / stride + 1;
        int W_out = (W_in + 2 * padding - kernelSize) / stride + 1;
        
        if (H_out <= 0 || W_out <= 0) {
            throw std::runtime_error("Convolution resulted in invalid output dimensions.");
        }

        Tensor3f out(numFilters, H_out, W_out);
        out.setZero();

        int channels_per_group = inChannels / groups;
        int filters_per_group = numFilters / groups;

        // Manual Forward Pass (highly parallelized)
        #pragma omp parallel for collapse(3)
        for (int f = 0; f < numFilters; ++f) {
            int group_idx = f / filters_per_group;
            
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    float sum = 0.0f;
                    
                    for (int c_in_g = 0; c_in_g < channels_per_group; ++c_in_g) {
                        int c_in = group_idx * channels_per_group + c_in_g;
                        
                        for (int ki = 0; ki < kernelSize; ++ki) {
                            for (int kj = 0; kj < kernelSize; ++kj) {
                                int in_i = i * stride + ki - padding;
                                int in_j = j * stride + kj - padding;

                                if (in_i >= 0 && in_i < H_in && in_j >= 0 && in_j < W_in) {
                                    sum += input(c_in, in_i, in_j) * kernels(f, c_in_g, ki, kj);
                                }
                            }
                        }
                    }
                    out(f, i, j) = sum;
                }
            }
        }
        
        int N = H_out * W_out;

        lastPreActivation = out;

        // Apply Bias and Activation
        #pragma omp parallel for
        for (int f = 0; f < numFilters; ++f) {
            Eigen::Map<Matrixf> mat(out.data() + f * N, H_out, W_out);
            mat.array() += biases[f];

            for (int i = 0; i < N; ++i) {
                float& val = out.data()[f * N + i];
                lastPreActivation.data()[f * N + i] = val; // Save value before activation
                switch(act){
                    case Activation::RELU: val = relu(val); break;
                    case Activation::GELU: val = gelu(val); break;
                    case Activation::SIGMOID: val = sigmoid(val); break;
                    case Activation::IDENTITY: break;
                }
            }
        }
        
        return out;
    }

    Tensor3f backward(const Tensor3f& dOut, float lr, int t){
        int n = lastInput.dimension(1); // Assuming square input H=W=n
        int outH = dOut.dimension(1);
        int outW = dOut.dimension(2);
        
        Tensor3f dInput(inChannels, n, n); dInput.setZero();

        int channels_per_group = inChannels / groups;
        int filters_per_group = numFilters / groups;

        Tensor4f dKernels(numFilters, channels_per_group, kernelSize, kernelSize); dKernels.setZero();
        Vectorf dBiases(numFilters); dBiases.setZero();

        #pragma omp parallel for
        for(int f=0; f<numFilters; f++){
            int group_idx = f / filters_per_group;

            for(int i=0; i<outH; i++){
                for(int j=0; j<outW; j++){
                    float grad = dOut(f, i, j);
                    float z = lastPreActivation(f, i, j);

                    // Apply Activation Derivative
                    switch(act){
                        case Activation::RELU: grad *= relu_deriv(z); break;
                        case Activation::GELU: grad *= gelu_deriv(z); break;
                        case Activation::SIGMOID: grad *= (sigmoid(z) * (1.0f - sigmoid(z))); break;
                        case Activation::IDENTITY: break;
                    }

                    dBiases[f] += grad;

                    // Compute dInput and dKernels
                    for(int c_in_g=0; c_in_g<channels_per_group; c_in_g++){
                        int c_in = group_idx * channels_per_group + c_in_g;

                        for(int ki=0; ki<kernelSize; ki++)
                            for(int kj=0; kj<kernelSize; kj++){
                                int in_i = i*stride + ki - padding;
                                int in_j = j*stride + kj - padding;
                                if(in_i>=0 && in_i<n && in_j>=0 && in_j<n){
                                    // Gradient for Input
                                    #pragma omp atomic
                                    dInput(c_in, in_i, in_j) += grad * kernels(f, c_in_g, ki, kj);

                                    // Gradient for Kernels
                                    dKernels(f, c_in_g, ki, kj) += grad * lastInput(c_in, in_i, in_j);
                                }
                            }
                    }
                }
            }
        }
        
        // Adam Update with Weight Decay (L2 Regularization)
        float lr_t = lr * std::sqrt(1.0f - std::pow(BETA2, t)) / (1.0f - std::pow(BETA1, t));

        for(int i=0; i<kernels.size(); ++i) {
            float dK = dKernels.data()[i];
            float K_val = kernels.data()[i];

            // Add Weight Decay (L2) to Gradient
            dK += WEIGHT_DECAY * K_val;

            // Adam Update
            M_kernels.data()[i] = BETA1 * M_kernels.data()[i] + (1.0f - BETA1) * dK;
            V_kernels.data()[i] = BETA2 * V_kernels.data()[i] + (1.0f - BETA2) * dK * dK;
            kernels.data()[i] -= lr_t * M_kernels.data()[i] / (std::sqrt(V_kernels.data()[i]) + EPSILON);
        }
        
        for(int i=0; i<numFilters; ++i) {
            float dB = dBiases[i];
            // Adam Update (Biases typically have no decay)
            M_biases[i] = BETA1 * M_biases[i] + (1.0f - BETA1) * dB;
            V_biases[i] = BETA2 * V_biases[i] + (1.0f - BETA2) * dB * dB;
            biases[i] -= lr_t * M_biases[i] / (std::sqrt(V_biases[i]) + EPSILON);
        }

        return dInput;
    }

    void save(const std::string& prefix) {
        ofstream ofs(prefix + ".bin", ios::binary);
        if (!ofs.is_open()) { cerr << "Cannot open " << prefix << ".bin for ConvLayer saving\n"; return; }

        int K_size = kernels.size();
        int B_size = biases.size();

        write_bin(ofs, &kernelSize, 1);
        write_bin(ofs, &numFilters, 1);
        write_bin(ofs, &inChannels, 1);
        write_bin(ofs, &groups, 1);

        write_bin(ofs, kernels.data(), K_size);
        write_bin(ofs, biases.data(), B_size);
    }
    
    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { cerr << "Cannot open " << prefix << ".bin for ConvLayer loading\n"; return; }

        int loaded_kSize, loaded_nFilters, loaded_inC, loaded_groups;
        read_bin(ifs, &loaded_kSize, 1);
        read_bin(ifs, &loaded_nFilters, 1);
        read_bin(ifs, &loaded_inC, 1);
        read_bin(ifs, &loaded_groups, 1);

        if (loaded_kSize != kernelSize || loaded_nFilters != numFilters || loaded_inC != inChannels || loaded_groups != groups) {
            cerr << "Configuration mismatch in ConvLayer " << prefix << endl;
            return;
        }

        read_bin(ifs, kernels.data(), kernels.size());
        read_bin(ifs, biases.data(), biases.size());
    }
};

// ---------------- ConvNeXt Block ----------------
struct ConvNeXtBlock {
    int channels;
    ConvLayer dwConv;
    LayerNorm ln;
    ConvLayer pwConv1; // Expansion
    ConvLayer pwConv2; // Projection

    ConvNeXtBlock(int C, int expansion_ratio = 4)
        : channels(C),
        // 7x7 Depthwise Conv, Padding 3, Stride 1, Groups=C
        dwConv(C, 7, C, Activation::IDENTITY, 1, 3, C),
        ln(C),
        // 1x1 Pointwise Expansion, uses GELU
        pwConv1(C, 1, C * expansion_ratio, Activation::GELU, 1, 0, 1),
        // 1x1 Pointwise Projection, uses IDENTITY
        pwConv2(C * expansion_ratio, 1, C, Activation::IDENTITY, 1, 0, 1)
    {}

    Tensor3f forward(const Tensor3f& input) {
        Tensor3f residual = input;

        Tensor3f x = dwConv.forward(input);
        x = ln.forward(x);
        x = pwConv1.forward(x);
        x = pwConv2.forward(x);

        // Residual connection
        x += residual;

        return x;
    }

    Tensor3f backward(const Tensor3f& dOut, float lr, int t) {
        Tensor3f dResidual = dOut;
        Tensor3f dBlock = dOut;

        // Backprop through MLP (pwConv2 -> pwConv1 -> ln)
        dBlock = pwConv2.backward(dBlock, lr, t);
        dBlock = pwConv1.backward(dBlock, lr, t);
        dBlock = ln.backward(dBlock, lr, t);
        
        // Backprop through Depthwise Conv
        dBlock = dwConv.backward(dBlock, lr, t);

        // Add residual gradient
        dBlock += dResidual;

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

// ---------------- Max Pooling ----------------
struct MaxPool {
    int poolSize;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::pair<int,int>>>> maxPos;
    int lastH, lastW, F;

    MaxPool(int poolSize_ = 2, int stride_ = 2, int padding_ = 0)
        : poolSize(poolSize_), stride(stride_), padding(padding_), lastH(0), lastW(0), F(0) {}

    Tensor3f forward(const Tensor3f& input)
    {
        F = input.dimension(0);
        int H = input.dimension(1);
        int W = input.dimension(2);
        lastH = H; lastW = W;

        int outH = (H + 2*padding - poolSize) / stride + 1;
        int outW = (W + 2*padding - poolSize) / stride + 1;

        maxPos.assign(F, std::vector<std::vector<std::pair<int,int>>>(outH, std::vector<std::pair<int,int>>(outW)));

        Tensor3f out(F, outH, outW);

        #pragma omp parallel for collapse(3)
        for(int f = 0; f < F; f++){
            for(int i = 0; i < outH; i++){
                for(int j = 0; j < outW; j++){
                    float m = -1e30f;
                    int max_i = -1, max_j = -1;
                    for(int pi = 0; pi < poolSize; pi++){
                        for(int pj = 0; pj < poolSize; pj++){
                            int in_i = i*stride + pi - padding;
                            int in_j = j*stride + pj - padding;
                            float val = (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) ? input(f, in_i, in_j) : -1e30f;
                            if(val > m){ m = val; max_i = in_i; max_j = in_j; }
                        }
                    }
                    out(f, i, j) = m;
                    maxPos[f][i][j] = {max_i, max_j};
                }
            }
        }
        return out;
    }

    Tensor3f backward(const Tensor3f& dOut)
    {
        int origH = lastH;
        int origW = lastW;

        Tensor3f dInput(F, origH, origW); dInput.setZero();

        #pragma omp parallel for collapse(3)
        for(int f = 0; f < F; f++){
            for(size_t i = 0; i < dOut.dimension(1); i++){
                for(size_t j = 0; j < dOut.dimension(2); j++){
                    int pi, pj;
                    std::tie(pi, pj) = maxPos[f][i][j];
                    if(pi >= 0 && pj >= 0 && pi < origH && pj < origW)
                        dInput(f, pi, pj) = dOut(f, i, j);
                }
            }
        }
        return dInput;
    }
    // No parameters to save/load
};

// ---------------- Fully Connected Layer ----------------
struct FCLayer {
    int inSize, outSize;
    Matrixf W;
    Vectorf B;
    
    // Adam moments
    Matrixf M_W, V_W;
    Vectorf M_B, V_B;

    // Backward pass state
    Vectorf lastIn;
    Vectorf lastPreActivation;
    Activation act;
    float dropoutProb;
    Vectorf dropoutMask;

    FCLayer(int inS, int outS, Activation act_ = Activation::SIGMOID, float dropout_=0.0f)
        : inSize(inS), outSize(outS), act(act_), dropoutProb(dropout_)
    {
        W.resize(outS, inS);
        B.resize(outS); B.setZero();
        M_W.resize(outS, inSize); M_W.setZero();
        V_W.resize(outS, inSize); V_W.setZero();
        M_B.resize(outS); M_B.setZero();
        V_B.resize(outS); V_B.setZero();
        
        std::mt19937 gen(std::random_device{}());
        for(int i=0; i<outS; i++)
            for(int j=0; j<inS; j++)
                W(i, j) = xavier_init(inS, outS);
    }

    Vectorf forward(const Vectorf& in, bool training=true){
        lastIn = in;
        Vectorf z = W * in + B;
        lastPreActivation = z;
        Vectorf out(outSize);
        dropoutMask.resize(outSize); dropoutMask.setOnes();

        for(int i=0; i<outSize; i++){
            float s = z(i);
            
            switch(act){
                case Activation::RELU: out(i) = relu(s); break;
                case Activation::GELU: out(i) = gelu(s); break;
                case Activation::SIGMOID: out(i) = sigmoid(s); break;
                case Activation::IDENTITY: out(i) = s; break;
            }

            if(training && dropoutProb > 0.0f){
                float scale = 1.0f / (1.0f - dropoutProb);
                float r = ((float)rand() / (float)RAND_MAX);
                if(r < dropoutProb){
                    out(i) = 0;
                    dropoutMask(i) = 0;
                } else {
                    out(i) *= scale;
                }
            }
        }
        return out;
    }

    Vectorf backward(const Vectorf& grad, float lr, int t){
        Vectorf dInput(inSize); dInput.setZero();
        // Matrixf dW(outSize, inSize); dW.setZero(); // Not needed as we update W directly
        // Vectorf dB(outSize); dB.setZero(); // Not needed as we update B directly

        // Adam learning rate with bias correction
        float lr_t = lr * std::sqrt(1.0f - std::pow(BETA2, t)) / (1.0f - std::pow(BETA1, t));

        #pragma omp parallel for
        for(int i=0; i<outSize; i++){
            float g = grad(i);

            if(dropoutMask(i) < 0.5f) {
                g = 0.0f;
            } else if (dropoutProb > 0.0f) {
                float scale = 1.0f / (1.0f - dropoutProb);
                g *= scale;
            }
            
            // Note: For SIGMOID output, the derivative is handled by BCE_grad, which returns (p-t).
            // For other activations, we apply the derivative here.
            if(act != Activation::SIGMOID) {
                float z = lastPreActivation(i);
                switch(act){
                    case Activation::RELU: g *= relu_deriv(z); break;
                    case Activation::GELU: g *= gelu_deriv(z); break;
                    case Activation::IDENTITY: g *= 1.0f; break;
                    case Activation::SIGMOID: break; 
                }
            }

            // Accumulate dB(i)
            float dB_val = g;
            // Adam Update for B(i) - no decay
            M_B(i) = BETA1 * M_B(i) + (1.0f - BETA1) * dB_val;
            V_B(i) = BETA2 * V_B(i) + (1.0f - BETA2) * dB_val * dB_val;
            B(i) -= lr_t * M_B(i) / (std::sqrt(V_B(i)) + EPSILON);

            for(int j=0; j<inSize; j++){
                // Calculate dW(i, j)
                float dWij = g * lastIn(j);
                
                // Accumulate dInput (needs atomic for shared j index)
                #pragma omp atomic
                dInput(j) += g * W(i, j);

                // Add Weight Decay (L2) to Gradient
                dWij += WEIGHT_DECAY * W(i, j);

                // Adam Update for W(i, j)
                M_W(i, j) = BETA1 * M_W(i, j) + (1.0f - BETA1) * dWij;
                V_W(i, j) = BETA2 * V_W(i, j) + (1.0f - BETA2) * dWij * dWij;
                W(i, j) -= lr_t * M_W(i, j) / (std::sqrt(V_W(i, j)) + EPSILON);
            }
        }
        
        return dInput;
    }

    void save(const std::string& prefix) {
        ofstream ofs(prefix + ".bin", ios::binary);
        if (!ofs.is_open()) { cerr << "Cannot open " << prefix << ".bin for FCLayer saving\n"; return; }

        int W_rows = W.rows();
        int W_cols = W.cols();
        int B_size = B.size();

        write_bin(ofs, &W_rows, 1);
        write_bin(ofs, &W_cols, 1);
        write_bin(ofs, &B_size, 1);

        write_bin(ofs, W.data(), W.size());
        write_bin(ofs, B.data(), B.size());
    }

    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { cerr << "Cannot open " << prefix << ".bin for FCLayer loading\n"; return; }

        int loaded_W_rows, loaded_W_cols, loaded_B_size;
        read_bin(ifs, &loaded_W_rows, 1);
        read_bin(ifs, &loaded_W_cols, 1);
        read_bin(ifs, &loaded_B_size, 1);
        
        if (loaded_W_rows != W.rows() || loaded_W_cols != W.cols() || loaded_B_size != B.size()) {
            cerr << "Weight size mismatch in FC layer " << prefix << endl;
            return;
        }

        read_bin(ifs, W.data(), W.size());
        read_bin(ifs, B.data(), B.size());
    }
};

// ---------------- Loss and Metrics ----------------

// Derivative of Binary Cross-Entropy (BCE) Loss with Sigmoid
// dL/dp = p - t
Vectorf BCE_grad(const Vectorf& p, const Vectorf& t){
    Vectorf g(p.size());
    for(int i=0;i<p.size();i++){
        // Clamp prediction to prevent log(0)
        float pi = std::clamp(p(i), 1e-7f, 1.0f-1e-7f);
        g(i) = pi - t(i); 
    }
    return g;
}

// Calculate multilabel accuracy (F1-score like metric based on threshold)
float multilabel_accuracy(const Vectorf& pred, const Vectorf& target, float threshold=0.5f){
    int correct=0, total=0;
    for(int i=0; i<pred.size(); i++){
        bool p = pred(i) > threshold;
        bool t = target(i) > 0.5f;
        if(p==t) correct++;
        total++;
    }
    return correct/(float)total;
}

// ---------------- Data Handling ----------------

struct Dataset {
    vector<Tensor3f> X;
    vector<Vectorf> Y;
    int N, C, H, W, num_labels;
};

// Function to load the pre-processed binary dataset
Dataset load_dataset_bin(const string& folder_path, map<string,int>& tag2idx) {
    Dataset data;
    string tag_file = folder_path + "/tags.json";
    ifstream ftag(tag_file);
    if(!ftag.is_open()){
        cerr << "Cannot open tag file " << tag_file << "\n";
        exit(1);
    }

    json j;
    ftag >> j;
    tag2idx.clear();
    for(auto &item: j.items())
        tag2idx[item.key()] = item.value();
    ftag.close();

    string bin_path = folder_path + "/dataset.bin";
    ifstream fin(bin_path, ios::binary);
    if(!fin.is_open()){
        cerr << "Cannot open dataset file " << bin_path << "\n";
        exit(1);
    }

    // Read header information
    read_bin(fin, &data.N, 1);
    read_bin(fin, &data.C, 1);
    read_bin(fin, &data.H, 1);
    read_bin(fin, &data.W, 1);
    read_bin(fin, &data.num_labels, 1);

    // Read data
    for(int i=0;i<data.N;i++){
        Tensor3f img(data.C, data.H, data.W);
        Vectorf labels(data.num_labels);
        read_bin(fin, img.data(), img.size());
        read_bin(fin, labels.data(), labels.size());
        data.X.push_back(std::move(img));
        data.Y.push_back(std::move(labels));
    }

    fin.close();
    cout << "Loaded " << data.N << " images of size "
         << data.C << "x" << data.H << "x" << data.W
         << " with " << data.num_labels << " labels.\n";

    return data;
}

void save_tags(const map<string,int>& tag2idx, const string& prefix){
    if(tag2idx.empty()){
        cerr << "[WARN] save_tags(): tag map is empty, nothing to save\n";
        return;
    }

    json j;
    for(const auto& p : tag2idx){
        j[p.first] = p.second;
    }

    string path = prefix + "_tags.json";
    ofstream f(path);
    if(!f.is_open()){
        cerr << "[ERROR] Failed to open tag file for writing: " << path << "\n";
        return;
    }

    try {
        f << j.dump(4);
    } catch(const json::exception& e){
        cerr << "[ERROR] JSON write failed: " << e.what() << "\n";
    }

    f.close();
    cout << "Saved tags: " << path << " (" << tag2idx.size() << " tags)\n";
}

void load_tags(map<string,int>& tag2idx, const string& prefix){
    string path = prefix + "_tags.json";
    ifstream f(path);
    if(!f.is_open()){
        cerr << "[ERROR] Cannot open tag file: " << path << "\n";
        return;
    }

    json j;
    try {
        f >> j;
    } catch(const json::exception& e){
        cerr << "[ERROR] Failed to parse tag JSON: " << e.what() << "\n";
        f.close();
        return;
    }

    if(!j.is_object()){
        cerr << "[ERROR] Tag file is not a JSON object: " << path << "\n";
        f.close();
        return;
    }

    tag2idx.clear();
    for(const auto& item : j.items()){
        if(!item.value().is_number_integer()){
            cerr << "[WARN] Skipping invalid tag entry: " << item.key() << "\n";
            continue;
        }
        tag2idx[item.key()] = item.value().get<int>();
    }

    if(tag2idx.empty()){
        cerr << "[WARN] Tag file loaded but contains no valid entries\n";
    } else {
        cout << "Loaded tags: " << tag2idx.size() << " tags\n";
    }

    f.close();
}

// ---------------- Training Helpers ----------------

// Helper function to save all model weights
void save_all_weights(const std::string& modelName,
                      ConvLayer& stem_conv, LayerNorm& stem_ln, ConvNeXtBlock& block1,
                      ConvLayer& expand2, ConvNeXtBlock& block2, ConvLayer& expand3,
                      ConvNeXtBlock& block3, ConvLayer& expand4, ConvNeXtBlock& block4,
                      FCLayer& classifier) {
    stem_conv.save(modelName + "_stem_conv");
    stem_ln.save(modelName + "_stem_ln");
    block1.save(modelName + "_block1");
    expand2.save(modelName + "_expand2");
    block2.save(modelName + "_block2");
    expand3.save(modelName + "_expand3");
    block3.save(modelName + "_block3");
    expand4.save(modelName + "_expand4");
    block4.save(modelName + "_block4");
    classifier.save(modelName + "_classifier");
}

// Helper function to load all model weights
void load_all_weights(const std::string& modelName,
                      ConvLayer& stem_conv, LayerNorm& stem_ln, ConvNeXtBlock& block1,
                      ConvLayer& expand2, ConvNeXtBlock& block2, ConvLayer& expand3,
                      ConvNeXtBlock& block3, ConvLayer& expand4, ConvNeXtBlock& block4,
                      FCLayer& classifier) {
    stem_conv.load(modelName + "_stem_conv");
    stem_ln.load(modelName + "_stem_ln");
    block1.load(modelName + "_block1");
    expand2.load(modelName + "_expand2");
    block2.load(modelName + "_block2");
    expand3.load(modelName + "_expand3");
    block3.load(modelName + "_block3");
    expand4.load(modelName + "_expand4");
    block4.load(modelName + "_block4");
    classifier.load(modelName + "_classifier");
}


// Function to run a full inference pass on a dataset partition (for evaluation)
std::pair<float, float> evaluate_model(
    const std::vector<Tensor3f>& X, const std::vector<Vectorf>& Y,
    ConvLayer& stem_conv, LayerNorm& stem_ln, ConvNeXtBlock& block1, MaxPool& downsample2,
    ConvLayer& expand2, ConvNeXtBlock& block2, MaxPool& downsample3, ConvLayer& expand3,
    ConvNeXtBlock& block3, MaxPool& downsample4, ConvLayer& expand4, ConvNeXtBlock& block4,
    GlobalAvgPool& gap, FCLayer& classifier) 
{
    float totalLoss = 0.0f;
    float totalAcc = 0.0f;
    size_t count = X.size();

    #pragma omp parallel for reduction(+:totalLoss, totalAcc)
    for(size_t i = 0; i < count; i++) {
        // Forward pass (Testing Mode)
        auto s = stem_conv.forward(X[i]);
        s = stem_ln.forward(s);
        s = block1.forward(s);
        s = downsample2.forward(s);
        s = expand2.forward(s);
        s = block2.forward(s);
        s = downsample3.forward(s);
        s = expand3.forward(s);
        s = block3.forward(s);
        s = downsample4.forward(s);
        s = expand4.forward(s);
        s = block4.forward(s);
        
        Vectorf pooled = gap.forward(s);
        Vectorf pred = classifier.forward(pooled, false); // Training=false (No dropout)
        
        // Loss calculation
        for(int k=0;k<pred.size();k++){
            float p_k = std::clamp(pred(k), 1e-7f, 1.0f-1e-7f);
            totalLoss += Y[i](k) > 0.5f ? -std::log(p_k) : -std::log(1.0f-p_k);
        }
        
        totalAcc += multilabel_accuracy(pred, Y[i]);
    }

    return {totalLoss / count, totalAcc / count};
}

// ---------------- Main Training Function ----------------

int main(int argc,char **argv){
    srand(time(0));
    random_device rd; mt19937 g(rd());

    map<string,int> tag2idx;

    // --- Training Configuration ---
    string dataFolder = "Data";
    string modelName = "model";
    string loadModelName = "";
    bool freeze_base = false;
    int epochs = 50;
    float base_lr = 0.0005f; 
    int batchSize = 2;
    int numThreads = 4;
    float warmup_epochs = 5.0f; // 5 epochs of linear warmup
    float val_ratio = 0.2f; // 20% for validation

    for(int i=1;i<argc;i++){
        string arg=argv[i];
        if(arg=="--data" && i+1<argc){ dataFolder = argv[i+1]; i++; }
        else if(arg=="--model-name" && i+1<argc){ modelName = argv[i+1]; i++; }
        else if(arg=="--epochs" && i+1<argc){ epochs = stoi(argv[i+1]); i++; }
        else if(arg=="--lr" && i+1<argc){ base_lr = stof(argv[i+1]); i++; }
        else if(arg=="--batch" && i+1<argc){ batchSize = stoi(argv[i+1]); i++; }
        else if(arg=="--threads" && i+1<argc){ numThreads = stoi(argv[i+1]); i++; }
        else if(arg=="--load-model" && i+1<argc){ loadModelName = argv[i+1]; i++; }
        else if(arg=="--freeze"){ freeze_base = true; }
        else if(arg=="--val-ratio" && i+1<argc){ val_ratio = stof(argv[i+1]); i++; }
    }

    omp_set_num_threads(numThreads);
    Eigen::setNbThreads(numThreads);

    cout << "Using " << numThreads << " OpenMP threads\n";
    cout << "Batch size set to: " << batchSize << endl;
    cout << "Base LR: " << base_lr << ", Warmup Epochs: " << warmup_epochs << ", Total Epochs: " << epochs << endl;

    cout << "Loading dataset from folder: " << dataFolder << "\n";
    Dataset full_data = load_dataset_bin(dataFolder, tag2idx);

    if (full_data.X.empty()) {
        cerr << "Error: No data loaded. Exiting.\n";
        return 1;
    }

    // --- Data Splitting ---
    vector<Tensor3f> train_X, val_X;
    vector<Vectorf> train_Y, val_Y;
    
    vector<int> all_indices(full_data.N);
    iota(all_indices.begin(), all_indices.end(), 0);
    shuffle(all_indices.begin(), all_indices.end(), g);

    size_t val_size = static_cast<size_t>(full_data.N * val_ratio);
    size_t train_size = full_data.N - val_size;
    
    cout << "Splitting data: Train size = " << train_size << ", Validation size = " << val_size << endl;

    for (size_t i = 0; i < full_data.N; ++i) {
        if (i < train_size) {
            train_X.push_back(std::move(full_data.X[all_indices[i]]));
            train_Y.push_back(std::move(full_data.Y[all_indices[i]]));
        } else {
            val_X.push_back(std::move(full_data.X[all_indices[i]]));
            val_Y.push_back(std::move(full_data.Y[all_indices[i]]));
        }
    }
    
    // Release full_data memory
    full_data.X.clear();
    full_data.Y.clear();


    // --- ConvNeXt Tiny-like Architecture Channels ---
    const int C1 = 64;
    const int C2 = 128;
    const int C3 = 256;
    const int C4 = 512;

    // --- Model Definition ---
    // 1. Stem: 4x4, stride 4
    ConvLayer stem_conv(full_data.C, 4, C1, Activation::IDENTITY, 4, 0, 1);
    LayerNorm stem_ln(C1);
    
    // 2. Stage 1 (Resolution / 4)
    ConvNeXtBlock block1(C1);

    // 3. Stage 2 Downsampling (Resolution / 8)
    MaxPool downsample2(2, 2);
    ConvLayer expand2(C1, 1, C2, Activation::IDENTITY, 1, 0, 1);
    ConvNeXtBlock block2(C2);

    // 4. Stage 3 Downsampling (Resolution / 16)
    MaxPool downsample3(2, 2);
    ConvLayer expand3(C2, 1, C3, Activation::IDENTITY, 1, 0, 1);
    ConvNeXtBlock block3(C3);

    // 5. Stage 4 Downsampling (Resolution / 32)
    MaxPool downsample4(2, 2);
    ConvLayer expand4(C3, 1, C4, Activation::IDENTITY, 1, 0, 1);
    ConvNeXtBlock block4(C4);

    // 6. Head
    GlobalAvgPool gap;
    int fcInputSize = C4;

    FCLayer classifier(fcInputSize, tag2idx.size(), Activation::SIGMOID, 0.2f); // Add 20% Dropout

    // --- Model Loading ---
    if (!loadModelName.empty()) {
        cout << "Loading model weights from " << loadModelName << "...\n";
        load_all_weights(loadModelName, stem_conv, stem_ln, block1, expand2, block2, expand3, block3, expand4, block4, classifier);
    }
    
    // --- Checkpointing variables ---
    float best_val_acc = -1.0f;


    // --- Training Loop ---
    long long global_step = 1;

    for(int e=0; e<epochs; e++){
        // Calculate Learning Rate using Cosine Annealing Schedule
        float lr_epoch = cosine_annealing_lr(base_lr, e, epochs, warmup_epochs);

        float totalLoss = 0;

        vector<int> indices(train_X.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), g);

        for(size_t i=0;i<train_X.size();i+=batchSize){
            for(int b=0;b<batchSize && i+b<train_X.size();b++){
                int idx = indices[i+b];

                // --- Forward Pass (Training Mode) ---
                auto s = stem_conv.forward(train_X[idx]);
                s = stem_ln.forward(s);
                s = block1.forward(s);
                s = downsample2.forward(s);
                s = expand2.forward(s);
                s = block2.forward(s);
                s = downsample3.forward(s);
                s = expand3.forward(s);
                s = block3.forward(s);
                s = downsample4.forward(s);
                s = expand4.forward(s);
                s = block4.forward(s);
                
                // Head
                Vectorf pooled = gap.forward(s);
                Vectorf pred = classifier.forward(pooled, true); // Training=true

                // --- Loss and Gradient Calculation ---
                Vectorf grad_fc = BCE_grad(pred, train_Y[idx]);
                
                // Accumulate Loss
                for(int k=0;k<pred.size();k++){
                    float p_k = std::clamp(pred(k), 1e-7f, 1.0f-1e-7f);
                    totalLoss += train_Y[idx](k) > 0.5f ? -std::log(p_k) : -std::log(1.0f-p_k);
                }
                
                // --- Backward Pass and Update ---
                int H = s.dimension(1);
                int W = s.dimension(2);
                
                global_step++; // Increment step for Adam's bias correction

                Vectorf dPooled = classifier.backward(grad_fc, lr_epoch, global_step);
                Tensor3f dConv = gap.backward(dPooled, H, W);

                if(!freeze_base) {
                    dConv = block4.backward(dConv, lr_epoch, global_step);
                    dConv = expand4.backward(dConv, lr_epoch, global_step);
                    dConv = downsample4.backward(dConv);

                    dConv = block3.backward(dConv, lr_epoch, global_step);
                    dConv = expand3.backward(dConv, lr_epoch, global_step);
                    dConv = downsample3.backward(dConv);

                    dConv = block2.backward(dConv, lr_epoch, global_step);
                    dConv = expand2.backward(dConv, lr_epoch, global_step);
                    dConv = downsample2.backward(dConv);

                    dConv = block1.backward(dConv, lr_epoch, global_step);
                    dConv = stem_ln.backward(dConv, lr_epoch, global_step);
                    dConv = stem_conv.backward(dConv, lr_epoch, global_step);
                }
            }
        }
        
        // --- End of Epoch Evaluation ---
        
        // 1. Evaluate on Training Set
        std::pair<float, float> train_metrics = evaluate_model(train_X, train_Y, stem_conv, stem_ln, block1, downsample2, expand2, block2, downsample3, expand3, block3, downsample4, expand4, block4, gap, classifier);
        float train_loss = train_metrics.first;
        float train_acc = train_metrics.second;

        // 2. Evaluate on Validation Set
        std::pair<float, float> val_metrics = evaluate_model(val_X, val_Y, stem_conv, stem_ln, block1, downsample2, expand2, block2, downsample3, expand3, block3, downsample4, expand4, block4, gap, classifier);
        float val_loss = val_metrics.first;
        float val_acc = val_metrics.second;

        // --- Print Epoch Results ---
        cout << "Epoch " << e+1 << "/" << epochs << ": LR = " << lr_epoch << " | Train Loss = " << train_loss 
             << ", Train Acc = " << train_acc << " | Val Loss = " << val_loss << ", Val Acc = " << val_acc << endl;
        
        if(freeze_base)
            cout << " [Incremental training: base frozen]" << endl;

        // 3. Checkpointing
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            cout << " *** New best validation accuracy: " << best_val_acc << ". Saving checkpoint: " << modelName << "_best.bin files ***" << endl;
            save_all_weights(modelName + "_best", stem_conv, stem_ln, block1, expand2, block2, expand3, block3, expand4, block4, classifier);
        }
    }

    // --- Save Final Model ---
    cout << "Saving final model as " << modelName << "_final.bin files...\n";
    save_all_weights(modelName + "_final", stem_conv, stem_ln, block1, expand2, block2, expand3, block3, expand4, block4, classifier);

    // --- After training loop ends ---
    cout << "Saving tags...\n";
    save_tags(tag2idx, modelName);

    return 0;
}
