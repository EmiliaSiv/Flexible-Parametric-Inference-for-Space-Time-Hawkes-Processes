import setuptools

setuptools.setup(
    name="spatio_temporal_fadin",
    version="0.0.1",
    description="Fast parametric inference for space-time Hawkes processes",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'scipy',
        'joblib',
        'matplotlib',
    ],
)