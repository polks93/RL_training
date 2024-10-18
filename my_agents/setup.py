from setuptools import setup, find_packages

setup(
    name='my_agents',
    version='0.2',
    packages=find_packages(),
    install_requires=['numpy', 'torch'],
    python_requires='>=3.6',
)