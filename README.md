# RNN Experiments

[![Build Status](https://travis-ci.com/jatinchowdhury18/RNNAudioEffects.svg?token=Ub9niJrqG1Br1qaaxp7E&branch=master)](https://travis-ci.com/jatinchowdhury18/RNNAudioEffects)

This repository contains code for training and implementing real-time
audio effects using single-sample recurrent neural networks. Work is
currently in progress, see below for a list of effects that will hopefully
be implemented:

**Effects:**
- [x] Hysteresis
- [ ] Phaser
- [ ] Reverse Distortion
- [ ] Restoration

Currently, RNN training is implemented using `Tensorflow`.
The trained RNNs are then loaded into audio plugins created with
the `JUCE` framework.

## Usage

To start, clone the repository and initialize submodules:
```bash
$ git clone https://github.com/jatinchowdhury18/RNNAudioEffects.git
$ cd RNNAudioEffects/
$ git submodule update --init --recursive
```

For each type of audio effect, there are `Python` scripts used to train
RNNs for that effect. These scripts are currently formated as
[VSCode `Python` notebooks](https://code.visualstudio.com/docs/python/jupyter-support-py), but can easily be converted to Jupyter notebooks,
or run as standalone `Python` scripts.

The audio plugin implementations of the RNNs are built using JUCE and
CMake. All JUCE-related dependencies are included in the repo as
submodules, but CMake must be installed by the user. As an example,
the hysteresis plugin can be built as follows:
```bash
$ ./setup.sh # set up submodules
$ cd hysteresis/Plugin
$ mkdir build $$ cd build/
$ cmake ..
$ cmake --build . --config Release
```

For computing the RNN output in real-time, several linear algebra
operations are required. This repo contains implementations using
both [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page),
a C++ linear algebra library, as well as the C++ Standard Library (STL).
The Eigen implementation can be enabled using the `USE_EIGEN` preprocessor
definition. If you would like to use the STL implementation (maybe for 
licensing reasons, or embedded applications), make sure this flag is
disabled (example from `hysteresis/Plugin/CMakeLists.txt`):
```cmake
# comment to use STL implementation instead of Eigen
add_definitions(-DUSE_EIGEN)
```

## About

I was very inspired by
[Alec Wright's paper](http://dafx2019.bcu.ac.uk/papers/DAFx2019_paper_43.pdf)
at last year's DAFx conference, where he proposes using recurrent neural
networks to model analog distortion circuits. This method is a marked 
improvement on existing neural net-based methods because of it's fast 
real-time performance, and since it allows for the inclusion of user 
controlled parameters to the model. Some limitations: in the work 
presented, only one control parameter is used; additionally, the audio 
processed by the neural network must be at the same sample
rate as the training data.

What I've been working on in this project is extending Wright's 
architecture to allow for multiple user-controlled parameters, and 
processing at variable sample rates. I've also been trying to extend the 
class of effects that can be created using RNNs, beyond distortion effects. 
So far I've successfully tested creating vibrato effects, and am working
to create phaser effects as well. I've also been experimenting with using
the RNN architecture to restore degraded or distorted audio.

---

## License

The code in this repository is freely available under the BSD 3-clause license. Enjoy!
