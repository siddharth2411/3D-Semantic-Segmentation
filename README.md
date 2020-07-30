# 3D Semantic segmentation

## Introduction

Semantic segmentation is the task of classifying each and every pixel in an image into a class.

semantic-8 is a benchmark for classification of 3D aerial data with 8 class labels, namely

1: man-made terrain

2: natural terrain

3: high vegetation

4: low vegetation

5: buildings

6: hardscape

7: scanning artifacts

8: cars

An additional label {0: unlabeled points} marks points without ground truth and should
not be used for training. In total over a billion points are provided.
Each data file consists of seven different components consisting of x,y,z coordinates,
intensity and R, G, B image value.

In this project, Open3D was used for
- Point cloud data loading and writing. Open3D provides efficient
  implementations of various point cloud manipulation methods.
- Data pre-processing, in particular, voxel-based down-sampling.
- Point cloud interpolation, in particular, fast nearest neighbor search for label
  interpolation.

## Procedure

### 1. Download the dataset

Download the dataset in a new folder named semantic_raw in dataset folder. from [Semantic3D](http://www.semantic3d.net): 

open the terminal
cd 3D Semantic segmentation/dataset/semantic_raw

├── bildstein_station1_xyz_intensity_rgb.labels 

├── bildstein_station1_xyz_intensity_rgb.txt

├── bildstein_station3_xyz_intensity_rgb.labels

├── bildstein_station3_xyz_intensity_rgb.txt

### 2. Convert the files to .pcd format using Open3D

Run
python preprocess.py
.pcd files will be generated in
3D Semantic segmentation/dataset/semantic_raw
├── bildstein_station1_xyz_intensity_rgb.labels

├── bildstein_station1_xyz_intensity_rgb.pcd (new)

├── bildstein_station1_xyz_intensity_rgb.txt

├── bildstein_station3_xyz_intensity_rgb.labels

├── bildstein_station3_xyz_intensity_rgb.pcd (new)

├── bildstein_station3_xyz_intensity_rgb.txt

### 3. Downsample the data

Run
python downsample.py

The downsampled dataset will be written to `dataset/semantic_downsampled`. Points with
label '0' are excluded during downsampling.

cd 3D Semantic segmentation/dataset/semantic_downsampled

├── bildstein_station1_xyz_intensity_rgb.labels

├── bildstein_station1_xyz_intensity_rgb.pcd

├── bildstein_station3_xyz_intensity_rgb.labels

├── bildstein_station3_xyz_intensity_rgb.pcd

### 4. Build custom tensorflow Operations
First, activate the virtualenv and make sure tensorflow can be found with current python.

python -c "import tensorflow as tf"

Then build TF ops. You'll need Cmake And Cuda.

Run the following commands in terminal
1. cd tf_ops
2. mkdir build
3. cd build
4. cmake ..
5. sudo make

After that the following .so files shall be in the build directory.

3D Semantic segmentation/tf_ops/build

├── libtf_grouping.so

├── libtf_interpolate.so

├── libtf_sampling.so

Now check if the kernals are working properly 
Run the following on terminal

1. cd 3D Semantic segmentation/tf_ops
2. python test_tf_ops.py

### 5. Train

Run
python train.py


By default, the training set will be used for training and the validation set
will be used for validation. To train with both training and validation set,
use the `--train_set=train_full` flag. Checkpoints will be output to
`log/semantic`.

### 6. Prediction

Specify the set on using which labels will be predicted, by defaulf it would take the validation set, for test set write set=test.
Mention the best model generated in log/semantic, for example the best model can be best_model_epoch_025.

Run 
1. cd..
2. python predict.py --ckpt log/semantic/best_model_epoch_025.ckpt \
                  --set=validation \
                  --num_samples=500

The prediction results will be written to 3D Semantic segmentation/result/sparse.

### 7. Interpolate the sparse data

To interpolate the sparse prediction to the full point cloud.

Run
python interpolate.py

The prediction results will be written to result/dense.

### 8. Visualisation

For visualisation use the raw .txt file and the interpolated lables file.
For example the file used is "untermaederbrunnen_station3_xyz_intensity_rgb"

Run
python visualise.py




