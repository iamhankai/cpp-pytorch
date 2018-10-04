## cpp-pytorch
PyTroch 1.0 preview - LOADING A PYTORCH MODEL IN C++
Details refer to: https://shiftlab.github.io/pytorch_tutorials/advanced/cpp_export.html

Main requires: PyTroch 1.0, opencv, cmake

### STEP 1
CONVERTING YOUR PYTORCH MODEL TO TORCH SCRIPT and SERIALIZING YOUR SCRIPT MODULE TO A FILE

- run `python tracing.py` and get `model.pt`

### STEP 2
LOADING YOUR SCRIPT MODULE IN C++ and EXECUTING

1. Download LibTorch [here](https://pytorch.org/) and unzip

2. Cmake
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/Users/hankai/code/cpp-pytorch/libtorch ..
make
```

3. Run demo
```
./exexample-app ../model.pt ../dog.png
```
