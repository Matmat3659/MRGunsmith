#include "crow_all.h"
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include "json.hpp"
#include <omp.h>
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

// --- Global Optimization Parameters (only for type definition consistency) ---
// Note: Adam moments (M_*, V_*) and decay constants are NOT needed for inference.
constexpr float BETA1 = 0.9f;
constexpr float BETA2 = 0.999f;
constexpr float EPSILON = 1e-8f;
constexpr float WEIGHT_DECAY = 0.05f;

// --- Enum and Utility from Trainer (Essential for Model Structure) ---
enum class Activation { GELU, RELU, SIGMOID, IDENTITY };

// Gaussian Error Linear Unit (GELU) Approximation
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// ReLU
inline float relu(float x) { return std::max(0.0f, x); }

// Sigmoid (for the final layer output)
inline float sigmoid(float x) {
    if (x < -20.0f) return 1e-7f;
    if (x > 20.0f) return 1.0f - 1e-7f;
    return 1.0f / (1.0f + std::exp(-x));
}

// ---------------- Binary I/O Utilities ----------------

template<typename T>
void read_bin(ifstream& ifs, T* data, size_t size) {
    if (size > 0) {
        ifs.read(reinterpret_cast<char*>(data), size * sizeof(T));
    }
}

// ---------------- Layer Normalization (Inference Version) ----------------
struct LayerNorm {
    int channels;
    Vectorf gamma; // Learnable scale
    Vectorf beta;  // Learnable shift

    // Adam moments removed

    LayerNorm(int C) : channels(C) {
        gamma.resize(C);
        beta.resize(C);
    }

    Tensor3f forward(const Tensor3f& input) {
        int H = input.dimension(1);
        int W = input.dimension(2);
        int N = H * W; // H*W elements per channel

        Tensor3f output(channels, H, W);
        const float epsilon = 1e-5f;

        for (int c = 0; c < channels; ++c) {
            Eigen::Map<const Matrixf> mat(input.data() + c * N, H, W);
            float mean = mat.mean();

            Matrixf diff = mat.array() - mean;
            float variance = diff.array().pow(2).mean();

            float invStdDev = 1.0f / std::sqrt(variance + epsilon);

            Matrixf normalized = diff.array() * invStdDev;

            Eigen::Map<Matrixf> out_mat(output.data() + c * N, H, W);
            out_mat = (normalized.array() * gamma[c] + beta[c]).matrix();
        }
        return output;
    }

    // No backward pass for inference

    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { throw std::runtime_error("Cannot open " + prefix + ".bin for LayerNorm loading"); }

        int C_size;
        read_bin(ifs, &C_size, 1);
        if (C_size != channels) {
            throw std::runtime_error("Channel size mismatch in LayerNorm " + prefix);
        }

        read_bin(ifs, gamma.data(), gamma.size());
        read_bin(ifs, beta.data(), beta.size());
    }
};

// ---------------- Convolution Layer (Inference Version) ----------------
struct ConvLayer {
    int kernelSize, numFilters, stride, padding;
    Activation act;
    int inChannels;
    int groups;

    Tensor4f kernels;
    Vectorf biases;

    // Adam moments removed

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
        biases.resize(numFilters);
    }

    Tensor3f forward(const Tensor3f& input) {
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

        // Manual Forward Pass (using OpenMP for acceleration)
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

        // Apply Bias and Activation
        #pragma omp parallel for
        for (int f = 0; f < numFilters; ++f) {
            Eigen::Map<Matrixf> mat(out.data() + f * N, H_out, W_out);
            mat.array() += biases[f];

            for (int i = 0; i < N; ++i) {
                float& val = out.data()[f * N + i];
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

    // No backward pass for inference

    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { throw std::runtime_error("Cannot open " + prefix + ".bin for ConvLayer loading"); }

        int loaded_kSize, loaded_nFilters, loaded_inC, loaded_groups;
        read_bin(ifs, &loaded_kSize, 1);
        read_bin(ifs, &loaded_nFilters, 1);
        read_bin(ifs, &loaded_inC, 1);
        read_bin(ifs, &loaded_groups, 1);

        if (loaded_kSize != kernelSize || loaded_nFilters != numFilters || loaded_inC != inChannels || loaded_groups != groups) {
            throw std::runtime_error("Configuration mismatch in ConvLayer " + prefix);
        }

        read_bin(ifs, kernels.data(), kernels.size());
        read_bin(ifs, biases.data(), biases.size());
    }
};

// ---------------- Max Pooling (Inference Version) ----------------
struct MaxPool {
    int poolSize;
    int stride;
    int padding;

    MaxPool(int poolSize_ = 2, int stride_ = 2, int padding_ = 0)
        : poolSize(poolSize_), stride(stride_), padding(padding_) {}

    Tensor3f forward(const Tensor3f& input)
    {
        int F = input.dimension(0);
        int H = input.dimension(1);
        int W = input.dimension(2);

        int outH = (H + 2*padding - poolSize) / stride + 1;
        int outW = (W + 2*padding - poolSize) / stride + 1;

        Tensor3f out(F, outH, outW);

        #pragma omp parallel for collapse(3)
        for(int f = 0; f < F; f++){
            for(int i = 0; i < outH; i++){
                for(int j = 0; j < outW; j++){
                    float m = -1e30f;
                    for(int pi = 0; pi < poolSize; pi++){
                        for(int pj = 0; pj < poolSize; pj++){
                            int in_i = i*stride + pi - padding;
                            int in_j = j*stride + pj - padding;
                            float val = (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) ? input(f, in_i, in_j) : -1e30f;
                            if(val > m){ m = val; }
                        }
                    }
                    out(f, i, j) = m;
                }
            }
        }
        return out;
    }
    // No backward pass or load/save for inference
};

// ---------------- Global Average Pooling (Inference Version) ----------------
struct GlobalAvgPool {
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
    // No backward pass or load/save for inference
};

// ---------------- Fully Connected Layer (Inference Version) ----------------
struct FCLayer {
    int inSize, outSize;
    Matrixf W;
    Vectorf B;

    // Adam moments and dropout removed
    Activation act;

    FCLayer(int inS, int outS, Activation act_ = Activation::SIGMOID, float dropout_=0.0f)
        : inSize(inS), outSize(outS), act(act_)
    {
        W.resize(outS, inS);
        B.resize(outS);
    }

    Vectorf forward(const Vectorf& in){ // Simplified forward pass for inference
        Vectorf z = W * in + B;
        Vectorf out(outSize);

        for(int i=0;i<outSize;i++){
            float s = z(i);
            switch(act){
                case Activation::RELU: out(i) = relu(s); break;
                case Activation::GELU: out(i) = gelu(s); break;
                case Activation::SIGMOID: out(i) = sigmoid(s); break;
                case Activation::IDENTITY: out(i) = s; break;
            }
            // Dropout is skipped for inference
        }
        return out;
    }

    // No backward pass for inference

    void load(const std::string& prefix) {
        ifstream ifs(prefix + ".bin", ios::binary);
        if (!ifs.is_open()) { throw std::runtime_error("Cannot open " + prefix + ".bin for FCLayer loading"); }

        int loaded_W_rows, loaded_W_cols, loaded_B_size;
        read_bin(ifs, &loaded_W_rows, 1);
        read_bin(ifs, &loaded_W_cols, 1);
        read_bin(ifs, &loaded_B_size, 1);

        if (loaded_W_rows != W.rows() || loaded_W_cols != W.cols() || loaded_B_size != B.size()) {
            throw std::runtime_error("Weight size mismatch in FC layer " + prefix);
        }

        read_bin(ifs, W.data(), W.size());
        read_bin(ifs, B.data(), B.size());
    }
};

// ---------------- ConvNeXt Block (Inference Version) ----------------
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

    // No backward pass for inference

    void load(const std::string& prefix) {
        dwConv.load(prefix + "_dw");
        ln.load(prefix + "_ln");
        pwConv1.load(prefix + "_pw1");
        pwConv2.load(prefix + "_pw2");
    }
};

// ---------------- Image Loading and Pre-processing ----------------

Tensor3f load_image_stb(const string& image_path, int target_H, int target_W) {
    int W_in, H_in, C;
    // Load image as float (0.0 to 1.0)
    float* img_data = stbi_loadf(image_path.c_str(), &W_in, &H_in, &C, 3);
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    if (C != 3) {
        stbi_image_free(img_data);
        throw std::runtime_error("Image must have 3 channels (RGB). Found: " + std::to_string(C));
    }

    cout << "Loaded image: " << image_path << " (" << W_in << "x" << H_in << "x" << C << ").\n";

    Tensor3f input_tensor(C, target_H, target_W);

    // Normalization parameters (standard ImageNet means/stds for ConvNeXt)
    const float means[] = {0.485f, 0.456f, 0.406f};
    const float stds[] = {0.229f, 0.224f, 0.225f};

    if (W_in != target_W || H_in != target_H) {
        cout << "Resizing image from " << H_in << "x" << W_in << " to " << target_H << "x" << target_W << " using Nearest Neighbor (NN).\n";
    }

    // Convert HWC to CHW, perform nearest-neighbor resizing, and normalize
    // This section replaces the original error-throwing size check.
    for(int c=0; c<C; c++) {
        for(int h_out=0; h_out<target_H; h_out++) {
            for(int w_out=0; w_out<target_W; w_out++) {
                // Nearest Neighbor Indexing for Resizing
                // Map output coordinate (h_out, w_out) to input coordinate (h_in, w_in)
                // This is a manual resize implementation.
                int h_in = (int)((float)h_out * H_in / target_H);
                int w_in = (int)((float)w_out * W_in / target_W);

                // Ensure indices are within bounds (shouldn't be necessary but good practice)
                h_in = std::min(h_in, H_in - 1);
                w_in = std::min(w_in, W_in - 1);

                // img_data is HWC, index is (h_in * W_in + w_in) * C + c
                float val = img_data[(h_in * W_in + w_in) * C + c];

                // Apply normalization: (value - mean) / std
                input_tensor(c, h_out, w_out) = (val - means[c]) / stds[c];
            }
        }
    }

    stbi_image_free(img_data);
    return input_tensor;
}

// ---------------- Tag Loading ----------------

map<int, string> load_tags(const string& model_dir, const string& model_name){
    string path = model_dir + "/" + "model_tags.json";
    ifstream f(path);
    if(!f.is_open()){
        throw std::runtime_error("Cannot open tag file: " + path);
    }

    json j;
    try {
        f >> j;
    } catch(const json::exception& e){
        f.close();
        throw std::runtime_error("Failed to parse tag JSON: " + string(e.what()));
    }

    if(!j.is_object()){
        f.close();
        throw std::runtime_error("Tag file is not a JSON object: " + path);
    }

    map<int, string> idx2tag;
    for(const auto& item : j.items()){
        if(!item.value().is_number_integer()){
            cerr << "[WARN] Skipping invalid tag entry: " << item.key() << "\n";
            continue;
        }
        idx2tag[item.value().get<int>()] = item.key();
    }

    if(idx2tag.empty()){
        throw std::runtime_error("Tag file loaded but contains no valid entries.");
    }

    cout << "Loaded " << idx2tag.size() << " tags.\n";
    return idx2tag;
}

std::mutex g_model_mutex;

// ---------------- Model Config ----------------
struct ModelConfig {
    std::string folder;
    std::string nickname;
    std::string checkpoint_prefix; // model_best, model_final, etc
};

std::map<std::string, ModelConfig> g_models;

// ---------------- Config Loader ----------------
void load_config(const std::string& path){
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open config: " + path);
    std::string line;
    while(std::getline(f, line)){
        if(line.empty() || line[0]=='#') continue;
        std::istringstream iss(line);
        std::string folder, nickname, prefix;
        if(!(iss >> folder >> nickname >> prefix)) continue;
        g_models[nickname] = {folder, nickname, prefix};
    }
    std::cout << "Loaded " << g_models.size() << " models from config.\n";
}

// ---------------- Inference Wrapper ----------------
json run_inference(const std::string& model_nick, const std::vector<std::string>& images, float threshold=0.5f, int threads=1){
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if(g_models.find(model_nick) == g_models.end())
        throw std::runtime_error("Model not found: " + model_nick);

    const auto& cfg = g_models[model_nick];
    const std::string modelDir = cfg.folder;
    const std::string prefix = modelDir + "/" + cfg.checkpoint_prefix; // FIXED

    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);

    // Load tags
    auto idx2tag = load_tags(modelDir, cfg.checkpoint_prefix);
    int num_labels = idx2tag.size();

    json results = json::array();

    for(auto& imagePath : images){
        Tensor3f input_tensor = load_image_stb(imagePath, 224, 224); // resize to 224x224

        // ---------------- Model Definition ----------------
        const int C_in = 3;
        const int C1 = 64, C2 = 128, C3 = 256, C4 = 512;

        ConvLayer stem_conv(C_in, 4, C1, Activation::IDENTITY, 4, 0, 1);
        LayerNorm stem_ln(C1);

        ConvNeXtBlock block1(C1);
        MaxPool downsample2(2,2);
        ConvLayer expand2(C1,1,C2); ConvNeXtBlock block2(C2);

        MaxPool downsample3(2,2);
        ConvLayer expand3(C2,1,C3); ConvNeXtBlock block3(C3);

        MaxPool downsample4(2,2);
        ConvLayer expand4(C3,1,C4); ConvNeXtBlock block4(C4);

        GlobalAvgPool gap;
        FCLayer classifier(C4, num_labels, Activation::SIGMOID);

        // Load weights
        stem_conv.load(prefix + "_stem_conv");
        stem_ln.load(prefix + "_stem_ln");
        block1.load(prefix + "_block1"); 
        expand2.load(prefix + "_expand2"); block2.load(prefix + "_block2");
        expand3.load(prefix + "_expand3"); block3.load(prefix + "_block3");
        expand4.load(prefix + "_expand4"); block4.load(prefix + "_block4");
        classifier.load(prefix + "_classifier");

        // ---------------- Forward Pass ----------------
        auto s = stem_conv.forward(input_tensor);
        s = stem_ln.forward(s);
        s = block1.forward(s); s = downsample2.forward(s); s = expand2.forward(s); s = block2.forward(s);
        s = downsample3.forward(s); s = expand3.forward(s); s = block3.forward(s);
        s = downsample4.forward(s); s = expand4.forward(s); s = block4.forward(s);

        Vectorf pooled = gap.forward(s);
        Vectorf pred = classifier.forward(pooled);

        // ---------------- Threshold & Collect Tags ----------------
        std::vector<std::string> tag_list;
        for(int i=0;i<pred.size();i++){
            if(idx2tag.count(i) && pred(i)>=threshold){
                tag_list.push_back(idx2tag[i]);
            }
        }

        results.push_back({
            {"Model", model_nick},
            {"Image", imagePath},
            {"Tags", tag_list}
        });
    }

    return results;
}

// ---------------- Main API Server ----------------
int main(int argc, char* argv[]){
    std::string config_file, address="127.0.0.1";
    uint16_t port=18080;
    int threads=1;

    for(int i=1;i<argc;i++){
        std::string arg = argv[i];
        if(arg=="--config" && i+1<argc) { config_file = argv[i+1]; i++; }
        else if(arg=="--port" && i+1<argc) { port = std::stoi(argv[i+1]); i++; }
        else if(arg=="--address" && i+1<argc) { address = argv[i+1]; i++; }
        else if(arg=="--threads" && i+1<argc) { threads = std::stoi(argv[i+1]); i++; }
    }

    if(config_file.empty()){
        std::cerr << "Error: --config <file> required\n"; return 1;
    }

    try { load_config(config_file); }
    catch(const std::exception& e){ std::cerr << "Config Error: " << e.what() << "\n"; return 1; }

    crow::App<crow::CORSHandler> app;
    app.get_middleware<crow::CORSHandler>().global(); // fixed for Crow

    // ---------------- /infer endpoint ----------------
    CROW_ROUTE(app,"/infer").methods("POST"_method)
    ([&](const crow::request& req){
        json req_json;
        try { req_json = json::parse(req.body); } 
        catch(...) { return crow::response(400, "Invalid JSON"); }

        if(!req_json.contains("Model") || !req_json.contains("Image"))
            return crow::response(400, "Missing Model or Image field");

        std::string model_nick = req_json["Model"];
        float threshold = req_json.value("threshold", 0.5f);
        std::vector<std::string> images;

        if(req_json["Image"].is_string()) images.push_back(req_json["Image"].get<std::string>());
        else if(req_json["Image"].is_array()){
            for(auto& v : req_json["Image"]) images.push_back(v.get<std::string>());
        } else return crow::response(400,"Invalid Image field");

        try {
            json res = run_inference(model_nick, images, threshold, threads);
            return crow::response(res.dump());
        } catch(const std::exception& e){
            return crow::response(500, std::string("Inference Error: ") + e.what());
        }
    });

    std::cout << "Server running at http://" << address << ":" << port << "\n";
    app.bindaddr(address).port(port).multithreaded().run();
    return 0;
}
