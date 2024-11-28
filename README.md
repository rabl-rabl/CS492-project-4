# Upsampling-Based Diffusion for 3D Voxel Generation

<div align=center>
  <h3>
  Doyoon Jeon, Junhyeok Yang, Sehyun Park
  </h3>
</div>


<p align="center"><img src="https://github.com/user-attachments/assets/acceed20-c532-4416-b12e-1dcfc797897b" width="400" height="600" /></p>


1. Download the dataset by load_data.py. You should put hdf5_data folder into data folder before running load_data.py.
```python
python load_data.py
```

2. Downsample the training data. Then you will get maxpool and avgpool npy file in hdf5_data folder.
```python
python downsampling.py --category chair|airplane|table
```

3. Train the DDPM model using downsampled data. It might take a few days for training.
```python
python train.py --category chair|airplane|table
```
After training, you will get intermediate samples and ckpt file in 'diffusion-voxel-(current_time)' folder.

4. Sample 1000x32x32x32 size numpy file using the ckpt file. You should change the checkpoint_path to get proper sample.
```python
python sample.py --category chair|airplane|table
```
After sampling, you will get {category}_combined_data.npy file in samples/{category} folder.
   
5. 

n. Train the upsampling network.
   1) In upscaling_network.py, change the path to the training data to select the category you want to train.
   2) Run upscaling_network.py