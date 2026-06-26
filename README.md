# README

This repo is my ML workspace - a collection of experiments as I learn ML.

File structure:
  * `src/`: all the code. All `.cpp` files in here are individual programs.
    - `src/fnn-demo.cpp`: demo of a simple fnn training run + inference.
  - `src/lib/`: the core library code - where the true ML content lives. Note: `src/lib/util.h` is important to keep in mind before contributing, it contains helpers that shorten code even if it's "bad practice".
    - `src/lib/util.h`: helpers
    - `src/lib/tensor.{h,cpp}`: tensor class
    - `src/lib/nn.{h,cpp}`: neural network class
    - `src/lib/autograd.{h,cpp}`: autograd
  * `ref/`: the reference notes for the impl.
    - `ref/autograd.tex`: autograd
    - `ref/crossentropy.tex`: proof of cross-entropy (via proper scoring rules)
  * `Makefile`: makefile.
