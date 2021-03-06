from setuptools import setup

setup(
    name= 'efsvm',
    version= '0.0.1',
    description= 'Implemenation of EFSVM',
    author= 'SUNGWOO HUR',
    author_email= 'hursungwoo@postech.ac.kr',
    url= 'https://github.com/mth9406/Entropy-Fuzzy-Support-Vector-Machine.git',
    install_requires= ['cvxopt == 1.2.7'],
    packages = ['efsvm'],
    zip_safe = False
)
