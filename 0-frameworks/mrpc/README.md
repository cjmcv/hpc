# mrpc

Mini-RPC

---

## 依赖 - Asio

Asio is a cross-platform C++ library for network and low-level I/O programming that provides developers with a consistent asynchronous model using a modern C++ approach.
* [Main Page](https://think-async.com/Asio/)
* [Github](https://github.com/chriskohlhoff/asio)

## 编译

```bash
cd hpc\0-frameworks\cux
mkdir windows
cd windows
cmake -G "Visual Studio 15 2017" -DCMAKE_GENERATOR_PLATFORM=x64 ..
```