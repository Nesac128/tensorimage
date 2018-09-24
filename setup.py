from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='nnir',
    version='1.0.0',
    description='Python package to train neural networks to classify data',
    long_description=long_description,
    author='PlanetGazer8360',
    author_email='ml.learner.8359@gmail.com',
    long_description_content_type="text/markdown",
    url="https://github.com/Nesac128/NNIR",
    install_requires=['tensorflow==1.9.0', 'Pillow==5.1.0', 'scikit-learn==0.19.1',
                      'numpy==1.14.5', 'opencv-python==3.4.0.12', 'pandas==0.22.0',
                      'click==6.7', 'matplotlib==2.0.2', 'scipy', 'png', 'flask',
                      'plotly', 'dash', 'dash_core_components', 'dash_html_components'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
