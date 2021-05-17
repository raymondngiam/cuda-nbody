## CUDA N-Body Simulation

---

### Dependencies

- CMake > 3.18
- CUDA > 11.1
- Boost > 1.71

### Pull third party submodules

```
$ cd <repo_root>
$ git submodule update --init --recursive
```

### Build

```
$ cd <repo_root>
$ mkdir build && cd build
$ cmake ..
$ make
```

### Run tests

```
$ cd <repo_root>/build
$ ctest -VV
```

### Run program

```
$ cd <repo_root>/bin
$ ./main
Hello from the CPU.
Hello also from the GPU.
```