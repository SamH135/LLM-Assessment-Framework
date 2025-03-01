from setuptools import setup, find_packages

setup(
    name="framework",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Existing ML/AI dependencies
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'numpy>=1.22.4,<1.23.0',  # Updated to be compatible with scipy
        'scipy>=1.7.3,<1.8.0',    # Added explicit scipy requirement
        'tokenizers>=0.13.0',
        'tqdm>=4.65.0',
        'datasets>=2.10.0',
        'huggingface-hub>=0.26.2',

        # API and web server dependencies
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'starlette>=0.14.2',
        'pydantic>=1.8.2',

        # CORS and HTTP requirements
        'python-multipart>=0.0.5',  # For handling form data
        'requests>=2.26.0',  # For making HTTP requests
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'black>=21.9b0',
            'flake8>=3.9.2',
            'isort>=5.9.3',
            'pylint>=2.5.2,<2.7',  # Added version constraint
            'astroid>=2.5.2,<2.7',  # Added to match pylint requirement
            'mccabe>=0.6.0,<0.7',   # Added to match pylint requirement
        ]
    },
    python_requires='>=3.8',
)