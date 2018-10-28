from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tensorimage',
    version='1.1.0',
    description='Machine learning image classification tool',
    long_description=long_description,
    author='TensorImage',
    author_email='tensor.image2@gmail.com',
    long_description_content_type="text/markdown",
    url="https://github.com/TensorImage/TensorImage",
    install_requires=['tensorflow==1.9.0', 'tensorboard', 'Pillow==5.1.0',
                      'scikit-learn==0.19.1', 'numpy==1.14.5', 'pandas==0.22.0',
                      'opencv-python==3.4.0.12', 'click==6.7', 'progress'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-2.0 License",
        "Operating System :: POSIX")
)
