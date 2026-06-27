# README

This repo is my ML playground - my own ML library "emel" plus demos that use it. All of `src/` is my work; `data/` is mostly GPT/Claude's work (fetching datasets & formatting them into my `.tensor` format).

File structure:
  * `src/`: demos that utilize the library. All `.cpp` files in here are individual programs.
    - `src/fnn-demo.cpp`: demo of a simple fnn training run + inference.
  - `src/emel/`: the core emel library code. Note: `src/emel/util.h` is important to keep in mind before contributing, it contains helpers that shorten code even if it's "bad practice".
    - `src/emel/util.h`: helpers
    - `src/emel/tensor.{h,cpp}`: tensor class
    - `src/emel/nn.{h,cpp}`: neural network class
    - `src/emel/autograd.{h,cpp}`: autograd
  * `ref/`: the reference notes for the impl.
    - `ref/autograd.tex`: autograd
    - `ref/crossentropy.tex`: proof of cross-entropy (via proper scoring rules)
  * `data/`: relevant training/test/etc data to be used by the code.
  * `Makefile`: makefile.
