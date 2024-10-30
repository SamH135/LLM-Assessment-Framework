from setuptools import setup, find_packages

setup(
    name="framework",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'tokenizers>=0.13.0',
        'tqdm>=4.65.0',
        'datasets>=2.10.0',
        'huggingface-hub>=0.16.0'
    ],
    python_requires='>=3.8',
)