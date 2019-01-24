[![Codacy Badge](https://api.codacy.com/project/badge/Grade/20ce98b051b94e048fdb47452aa334c5)](https://app.codacy.com/app/TensorImage/tensorimage?utm_source=github.com&utm_medium=referral&utm_content=TensorImage/tensorimage&utm_campaign=Badge_Grade_Dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/tensorimage/tensorimage.svg)](https://GitHub.com/tensorimage/tensorimage/releases/)

# TensorImage
![Example](https://cdn-images-1.medium.com/max/1600/1*PAqzvCxPjpDN8RC9HQw45w.jpeg) 


__TensorImage__ is an open source library designed to make training and deploying image classification models easy.

## Features
- Cluster training: automatically compare the performance of multiple trainers, speeding up the process of hyperparameter tuning and feature engineering, as there is no need to do it manually

- Multithreaded training: by default, all training operations are run in 10 threads to make training models faster

- Built-in image data augmentation operations, which can be used for feature engineering:
    - Image flipping
    - Salt-pepper noise
    - Random brightness
    - Random contrast
    - Random hue
    - Random saturation
    - Gaussian blur
    - Colour filtering

- Workspace organization: all datasets, trained models, and internal metadata files are stored automatically inside a workspace directory, where you can quickly find any files you need

- Large-scale image classification: deploy trained models on thousands of images, with predictions for all images being stored in your workspace directory

## Upcoming features
- More data augmentation operations to apply on images:
    - Affine/perspective transformations
    - Random zooming
    - Random cropping
    - Individual pepper and salt noise
    - More image blurring techniques:
        - Median blur
        - Average blur
        - Motion blur
        - Bilateral blur
    - Translation
    
    
- Option to apply different data augmentation operations at once, e.g: instead of only applying gaussian blur, to be able to apply gaussian blur, pepper salt noise and random contrast at once, not uniquely separately

- Model inference for individual/batches of images for real-time prediction without writing on disk

- Real-time training from individual/batches of images without reading from disk, automatically training the model from new data, linked to real-time inference without having to store the model in disk (with option to store available) 

## Installation
From the terminal:
```shell
$ pip3 install tensorimage
```

## Documentation
You can view TensorImage's documentation [here](https://tensorimage.readthedocs.io/en/latest/).

## Support
If you are experiencing any errors or bugs, please report them in the [issues section](https://github.com/TensorImage/TensorImage/issues) or contact us at tensor.image2@gmail.com

## Contributing
If you have any ideas for features that should be added to TensorImage, please feel free to [fork](https://github.com/TensorImage/tensorimage/network/members) TensorImage and [open a pull request](https://github.com/TensorImage/tensorimage/pulls).

## License
TensorImage is licensed under the [MIT](https://github.com/TensorImage/tensorimage/blob/master/LICENSE.md) license.
