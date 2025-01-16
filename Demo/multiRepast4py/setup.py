from setuptools import setup, find_packages

setup(
    name='multiRepast4py',  
    version='0.1',  # Initial version number
    description='A package for creating and managing multilayer graphs',  # Brief description of the package
    author='Keng-Lien Lin',  
    author_email='kenglienl@gmail.com',  
    url='https://github.com/KengLL/MultiRepast4py',  
    packages=find_packages(),  # Automatically find and include all packages (subdirectories with __init__.py)
    install_requires=[
        'multinetx',  # List of dependencies
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],
    keywords='multilayer graph network multinetx',  # Keywords for easier searching
)
