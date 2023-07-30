# Windows Native (Python 3.9 or above)

**Step 1: Install Miniconda**

Download the latest Miniconda installer for Windows (64-bit) from the link below and run the installer.

(Miniconda document Website)[https://docs.conda.io]
(Miniconda Installer)[https://docs.conda.io/en/latest/miniconda.html]

Window
- 32bit Miniconda3 Windows 32-bit
- 64bit Miniconda3 Windows 64-bit

**Step 2: Create and Activate Conda Environment**

Open a command prompt or Anaconda prompt and create a new conda environment named "tf" with Python 3.9 by running the following commands:

```
conda create --name tf python=3.9
conda activate tf
```

**Step 3: Install Necessary Packages**

Upgrade pip and install TensorFlow and Pillow by running the following commands:

```
pip install --upgrade pip
pip install tensorflow
pip install pillow
```

# Windows with WSL (Python 3.9 or above)

**Step 1: Install Miniconda**

Open a WSL terminal and use the following commands to download and run the Miniconda installer for Linux:

```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2: Create and Activate Conda Environment**

In the WSL terminal, create a new conda environment named "tf" with Python 3.9 and activate it using the following commands:

```
conda create --name tf python=3.9
conda activate tf
```

**Step 3: Install Necessary Packages**

For GPU Support:

```
conda install cuda -c nvidia/label/cuda-11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install --upgrade pip
pip install tensorflow==2.12.*
pip install pillow
```

Followed by:

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If you have installed everything correctly, it should display:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Without GPU Support:

```
pip install --upgrade pip
pip install tensorflow==2.12.*
pip install pillow
```

Verify if TensorFlow is Ready:
To ensure TensorFlow is installed correctly, run the following command:

```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

It should display output like:

```
tf.Tensor(-992.188, shape=(), dtype=float32)
```

or

```
tf.Tensor(108.81939, shape=(), dtype=float32)
```

The specific number doesn't matter; this command just checks if TensorFlow is installed correctly by generating random numbers.

These steps should help you set up a Conda environment with TensorFlow on both Windows Native and Windows with WSL.