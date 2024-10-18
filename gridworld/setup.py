from setuptools import setup, find_packages

setup(
    name='gridworld',
    version='0.1',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy', 'pygame'],
    description='A custom GridWorld environment for gymnasium',
    author='Il tuo nome',
    author_email='tuo_email@example.com',
    url='https://github.com/tuo_username/gridworld_env',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)