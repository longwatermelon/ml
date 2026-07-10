# README

This repo is my ML playground - my own ML library "emel" plus demos that use it. All of `src/` is my work; `data/` and `tests/` is mostly GPT/Claude's work (fetching datasets & formatting them into my `.tensor` format, and writing tests to check emel's correctness).

File structure:
  * `src/`: demos that utilize the library. All `.cpp` files in here are individual programs.
    - `src/README.md`: docs on what each `src/*.cpp` example is demonstrating.
  - `src/emel/`: the core emel library code.
    - `src/emel/util.h`: helpers that shorten the code (important!).
    - `src/emel/tensor.{h,cpp}`: tensor class
    - `src/emel/nn.{h,cpp}`: neural network class
    - `src/emel/autograd.{h,cpp}`: autograd
    - `src/emel/opt.{h,cpp}`: gradient descent optimizers
    - `src/emel/tokenizer.{h,cpp}`: tokenizers
  * `ref/`: the reference notes for the impl.
    - `ref/autograd.tex`: autograd
    - `ref/crossentropy.tex`: proof of cross-entropy (via proper scoring rules)
    - `ref/conv2d.tex`: notes on implementing the conv2d layer
    - `ref/transformer.tex`: attention/mha/transformers
    - `ref/tf2gpt.tex`: from transformers to gpt
  * `data/`: relevant training/test/etc data to be used by the code
  * `tests/`: library tests
  * `Makefile`: makefile
