# Windows Native (Python 3.9 or above)

Step 1: Install Miniconda
Download the latest Miniconda installer for Windows (64-bit) from the link below and run the installer.
Miniconda Installer

Step 2: Create and Activate Conda Environment
Open a command prompt or Anaconda prompt and create a new conda environment named "tf" with Python 3.9 by running the following commands:

```
conda create --name tf python=3.9
conda activate tf
```

Step 3: Install Necessary Packages
Upgrade pip and install TensorFlow and Pillow by running the following commands:

```
pip install --upgrade pip
pip install tensorflow pillow
```

# Windows with WSL (Python 3.9 or above)

Step 1: Install Miniconda
Open a WSL terminal and use the following commands to download and run the Miniconda installer for Linux:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Step 2: Create and Activate Conda Environment
In the WSL terminal, create a new conda environment named "tf" with Python 3.9 and activate it using the following commands:

```
conda create --name tf python=3.9
conda activate tf
```

Step 3: Install Necessary Packages

3.1 # For GPU Support:
```
conda install cuda -c nvidia/label/cuda-11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install --upgrade pip
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

3.2 # Without GPU Support:
```
pip install --upgrade pip
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

These steps should help you set up a Conda environment with TensorFlow on both Windows Native and Windows with WSL.