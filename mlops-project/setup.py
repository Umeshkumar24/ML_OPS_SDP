from setuptools import setup, find_packages

setup(
    name='mlops-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A basic MLOps project with a pipeline workflow.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'jupyter',
        'matplotlib',
        'seaborn',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)