from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='TensorImage',
    version='2.0.1',
    description='Image classification library for easily and quickly deploying models and training classifiers',
    long_description=long_description,
    author='TensorImage',
    author_email='tensor.image2@gmail.com',
    long_description_content_type="text/markdown",
    url="https://github.com/TensorImage/TensorImage",
    install_requires=['tensorflow>=1.9.0', 'tensorboard', 'Pillow>=5.1.0',
                      'scikit-learn>=0.19.1', 'numpy>=1.14.5', 'pandas>=0.22.0',
                      'opencv-python>=3.4.0.12', 'progress', 'scipy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX"),
    packages=find_packages(),
    include_package_data=True,
)
