# MRGunsmith
A ConvNeXt like trainer and runner in cpp
THIS PROJECT IS UNDER DEVELOPMENT AND MAY CONTAIN BUGS

Required
  -  Eigen
  -  crow_all.h specificly in this folder

Compile guide

for the trainer
``` trainer
clang++ -fopenmp -O3 trainer.cpp -o trn -fopenmp
```
and for the runner
``` runner
clang++ -fopenmp -O3 interferer.cpp -o runner -fopenmp
```
basic usage's

prepair a dataset as sutch, this is what i usually do but you may automate it
make a folder with this structure, and a tags.json

data/
 - i1/
 - i2/
 - tags.json
 - and more if you want to

inside the tags.json would be

i1 is the folder name, or any as long as it matches the one in the json, the amount of tags can be any
same with i2
``` json example
{
	"i1": {
		"image.png": ["tags", "tags2", "tags3"],
		"image2.jpg": ["tags", "tags2", "tags3"]
	},
	"i2": {
		"image.png": ["tags", "tags2", "tags3"],
		"image2.jpeg": ["tags", "tags2", "tags3"]
	}
}
```

the trainer can be started with
```
./trn --epochs n --lr n --data path/to/the/dataset
```
but i recommend making a dedicated folder, then calling trn there, since it will output a lot of .bin
weights whitch can get overwhelming, like

```
./preprocessor <in> <out> <size>
```
or
```
./preprocessor folder/dataset folder/datasetout 256
```
the dataset is still the same, its just preprocessed

```
mkdir model1
cd model1
../trn --epoch n --lr n -data --path path/to/the/dataset
```

flags thingy
```
--data
```
sets where the folder where the dataset.bin and tags.json are in that was outputted by the preprocessor
```
--epoch
```
sets the epoch of the training
```
--lr
```
sets the learnrate of the training, usually 0.0001 or so is good for small dataset's
```
--batch
```
sets the batch size, 2 is default, please dont put this higher than the your dataset
```
--threads
```
sets threads, default is 4
```
--size
```
usual is 128x128, can be increades or decreased
```
--load-model
```
used if you have a already trained model, and using that model to train it with a newer dataset

for the runner is just
```
./runner path/to/image --model_folder path/to/where/the/model/is
```

server api
input
  -  single image
  ```
  {
    "Model": "modlename",
    "Image": "path/to/image",
    "threshold" 0.0 to 1.0
  }
  ```
  -  batch images
  ```
  {
  	"Model": "modelname",
  	"Image": ["pathtoimage1", "pathtoimage2", "pathtoimage3", ect],
  	"threshold": 0.0 to 1.0
  }
  ```
server commands
```
--config
--port
--address
--threads
```
server --config structure
```
Modelfolder/path modelnickname modlebase(eg the model_best or model_final)
Ex.
MD19/SUBMD19 MD19 model_final
```

returns
```
{
     "Image":"../1.jpg",
     "Model":"MD19",
     "Tags":["illustrated","anime","anime_girl"]
}
```

please give a star! i put a lot of time and stupidity on this!
