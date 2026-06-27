# README

This repo is my ML playground - my own ML library plus demos that use it.

File structure:
  * `src/`: demos that utilize the library. All `.cpp` files in here are individual programs.
    - `src/fnn-demo.cpp`: demo of a simple fnn training run + inference.
  - `src/lib/`: the core library code. Note: `src/lib/util.h` is important to keep in mind before contributing, it contains helpers that shorten code even if it's "bad practice".
    - `src/lib/util.h`: helpers
    - `src/lib/tensor.{h,cpp}`: tensor class
    - `src/lib/nn.{h,cpp}`: neural network class
    - `src/lib/autograd.{h,cpp}`: autograd
  * `ref/`: the reference notes for the impl.
    - `ref/autograd.tex`: autograd
    - `ref/crossentropy.tex`: proof of cross-entropy (via proper scoring rules)
  * `data/`: relevant training/test/etc data to be used by the code.
  * `Makefile`: makefile.
