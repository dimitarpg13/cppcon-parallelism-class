# Parallelism in Modern C++; from CPU to GPU
### Exercise 4: Image grayscale

---

In this first exercise you will learn:
* How to create a simple image processing kernel that performs a grayscaling.
* How to linearize the work-item id in row-major and column major ordering and how this affects global memory access coalescing.
* How to vectorize kernel code using SYCL vector types.

---

An image can be grayscaled using the following algorithm: 

```
Y = (R * 0.229) + (G * 0.587) + (B * 0.114)
```

For the purposes of this exercise the STB image loading and writing library has been made available and the source for this exercise already contains the appropriate API calls to get you started.

1.) Write a SYCL kernel function for performing a grayscaling

The source for this example provides a stub which loads and write an image using the STB image library. You will need to fill in the SYCL code for performing a grayscaling algorithm on the image using what you have learnt in the previous exercises.

The source also contains a call to a benchmarking utility that will print the time taken to execute the SYCL code, the SYCL code should go inside the lambda that is passed to the `benchmark` function.

+ Change the path to the image, feel free to use your own image, but be wary of the size.
+ Image loaded has four channels (RGBA)
+ It's recommended that you use a 2 dimensional range for your kernel, but a 1 dimensional range for you buffer.

2.) Evaluate global memory access

Now that you have a working grayscaling kernel you should evaluate whether the global memory access patterns in your kernel are coalesced.

Consider two alternative ways to linearise the global id:

```
auto rowMajorLinearId    = (idx[1] * width) + idx[0];  // row-major
auto columnMajorLinearId = (idx[0] * height) + idx[1];  // column-major
```

Try using both of these and compare the execution time of each.

3.) Use vectorisation

Now that global memory access is coalesced another optimization you could do here would be to use SYCL vectors to present the pixels in the image.

You can reinterpret a buffer to be represented as a different type using the `buffer` class' `reinterpret` member function template. When calling this function you must specify the new type as a template parameter and a new `range` that will represent elements of the new type within the same space in memory as a function parameter.

Try reinterpreting your buffer to use `cl::sycl::float4` instead of `float`.