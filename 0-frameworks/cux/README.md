# cux

An eXperimental framework for performance analysis and optimization of CUDA kernel functions.

This framework is designed to help you intuitively know the optimizable space of your own kernel functions, making it easier for you to make further optimizations.

---

## Build

```bash
cd hpc\0-frameworks\cux
mkdir windows
cd windows
cmake -G "Visual Studio 14 2015" -DCMAKE_GENERATOR_PLATFORM=x64 ..
```