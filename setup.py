from setuptools import setup, find_packages

setup(
    name='ood-with-vit',
    version='1.0.0',
    author='Simcs',
    author_email='smh3946@gmail.com',
    url='https://github.com/Simcs/OOD-with-ViT.git',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only'
    ],
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchmetrics==0.5.1',
        'torchvision',
        'scikit-image',
        'matplotlib==3.4.3',
        'timm==0.4.12',
    ],
    python_requires='>=3.8',
)