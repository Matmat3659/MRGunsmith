#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

// ---------------- Save tags ----------------
void save_tags(const map<string,int>& tag2idx, const string& folder){
    json j;
    for(auto &p: tag2idx) j[p.first] = p.second;
    ofstream f(folder + "/tags.json");
    f << j.dump(4);
    f.close();
}

// ---------------- Load image with resize ----------------
vector<vector<vector<float>>> load_image_resized(string path, int targetSize){
    int w, h, c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if(!img){ cerr << "Failed to load " << path << endl; exit(1); }

    vector<vector<vector<float>>> out(3, vector<vector<float>>(targetSize, vector<float>(targetSize, 0)));
    const float inv255 = 1.0f / 255.0f;

    for(int i=0; i<targetSize; i++){
        float fx = i * (h-1.0f)/(targetSize-1.0f);
        int x0 = (int)fx, x1 = min(x0+1, h-1);
        float dx = fx - x0;
        for(int j=0; j<targetSize; j++){
            float fy = j * (w-1.0f)/(targetSize-1.0f);
            int y0 = (int)fy, y1 = min(y0+1, w-1);
            float dy = fy - y0;

            for(int ch=0; ch<3; ch++){
                float v00 = img[(x0*w + y0)*3 + ch]*inv255;
                float v01 = img[(x0*w + y1)*3 + ch]*inv255;
                float v10 = img[(x1*w + y0)*3 + ch]*inv255;
                float v11 = img[(x1*w + y1)*3 + ch]*inv255;
                out[ch][i][j] = (1-dx)*(1-dy)*v00 + (1-dx)*dy*v01 + dx*(1-dy)*v10 + dx*dy*v11;
            }
        }
    }

    for(int ch=0; ch<3; ch++)
        for(int i=0; i<targetSize; i++)
            for(int j=0; j<targetSize; j++)
                out[ch][i][j] = (out[ch][i][j] - 0.5f) / 0.5f;

    stbi_image_free(img);
    return out;
}

// ---------------- Augment image ----------------
vector<vector<vector<float>>> augment_image(const vector<vector<vector<float>>>& img){
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    auto out = img;
    int C = img.size();
    int H = img[0].size();
    int W = img[0][0].size();

    // Horizontal flip
    if (uniform(rng) < 0.5f){
        for (int c=0; c<C; c++)
            for(int h=0; h<H; h++)
                for(int w=0; w<W/2; w++)
                    std::swap(out[c][h][w], out[c][h][W-1-w]);
    }

    // Rotate 90 degrees clockwise
    if(uniform(rng) < 0.5f){
        vector<vector<vector<float>>> rotated(C, vector<vector<float>>(W, vector<float>(H)));
        for(int c=0; c<C; c++)
            for(int h=0; h<H; h++)
                for(int w=0; w<W; w++)
                    rotated[c][w][H-1-h] = out[c][h][w];
        out = std::move(rotated);
    }

    // Color jitter
    float brightness = 0.2f;
    float contrast = 0.2f;
    float b = (uniform(rng) * 2 - 1) * brightness;
    float c = 1.0f + (uniform(rng) * 2 - 1) * contrast;

    for(int ch=0; ch<C; ch++)
        for(int h=0; h<H; h++)
            for(int w=0; w<W; w++){
                float val = out[ch][h][w] * c + b;
                out[ch][h][w] = std::min(std::max(val, -1.0f), 1.0f);
            }

    return out;
}

// ---------------- Preprocessor main ----------------
int main(int argc, char** argv){
    if(argc < 3){
        cerr << "Usage: " << argv[0] << " <input_folder> <output_folder> <target_size>\n";
        return 1;
    }

    string folder = argv[1];
    string out_folder = argv[2];
    int targetSize = stoi(argv[3]);

    map<string,int> tag2idx;

    // Load tags.json
    ifstream tfile(folder + "/tags.json");
    if(!tfile.is_open()){ cerr<<"Cannot open tags.json\n"; return 1; }
    json j; tfile >> j;

    // Build tag2idx map
    int numTags = 0;
    int total_images = 0;
    for(auto &f:j.items())
        for(auto &img:f.value().items()){
            for(auto &tag:img.value())
                if(tag2idx.find(tag) == tag2idx.end())
                    tag2idx[tag] = numTags++;
            total_images++;
        }

    cout << "Found " << total_images << " images and " << tag2idx.size() << " tags.\n";
    save_tags(tag2idx, out_folder);

    // Open binary file
    ofstream fout(out_folder + "/dataset.bin", ios::binary);
    if(!fout.is_open()){ cerr<<"Cannot open output file\n"; return 1; }

    // Write header: N, C, H, W, num_labels
    int N = total_images;
    int C = 3;
    int H = targetSize;
    int W = targetSize;
    int num_labels = tag2idx.size();
    fout.write((char*)&N, sizeof(int));
    fout.write((char*)&C, sizeof(int));
    fout.write((char*)&H, sizeof(int));
    fout.write((char*)&W, sizeof(int));
    fout.write((char*)&num_labels, sizeof(int));

    // Stream each image
    int count = 0;
    for(auto &f:j.items()){
        string subfolder = folder + "/" + f.key();
        for(auto &img:f.value().items()){
            string path = subfolder + "/" + img.key();
            auto imgRGB = load_image_resized(path, targetSize);
            imgRGB = augment_image(imgRGB);

            // Write image
            for(int c=0; c<C; c++)
                for(int h=0; h<H; h++)
                    for(int w=0; w<W; w++){
                        float val = imgRGB[c][h][w];
                        fout.write((char*)&val, sizeof(float));
                    }

            // Write labels
            vector<float> multi(num_labels, 0.0f);
            for(auto &tag: img.value())
                multi[tag2idx[tag]] = 1.0f;
            for(int l=0; l<num_labels; l++)
                fout.write((char*)&multi[l], sizeof(float));

            count++;
            if(count % 50 == 0) cout << "\rProcessed " << count << "/" << N << " images" << flush;
        }
    }

    fout.close();
    cout << "\nDone! Saved " << out_folder << "/dataset.bin and " << out_folder << "/tags.json\n";
    return 0;
}
