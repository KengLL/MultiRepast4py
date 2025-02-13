from setuptools import setup, find_packages

setup(
    name='multirepast4py',  
    version='0.1.1',      
    description='A Python package for multilayer network simulations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KengLL/MultiRepast4py',  
    author='Keng-Lien Lin',
    author_email='kenglienl@gmail.com',
    license='MIT', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    install_requires=[],   
    python_requires='>=3.8',
)
