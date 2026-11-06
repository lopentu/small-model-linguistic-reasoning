# Optional parallel driver

This source belongs inside a llama.cpp b9964 checkout, where `llama-common`
is defined. It is not a standalone CMake project.

```bash
git clone --branch b9964 --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
cp -r dev/iolbatch /path/to/llama.cpp/examples/iolbatch
```

Add `add_subdirectory(iolbatch)` to that checkout's `examples/CMakeLists.txt`, then:

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build /path/to/llama.cpp/build --target llama-iolbatch -j
```

Select hardware backend options for your system when configuring CMake.
This optional build is not covered by CPU Python checks. Preserve `LICENSE`
with distributed source and binaries.
