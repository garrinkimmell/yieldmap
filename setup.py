from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt')

setup(
    name='YieldMap',
    version='0.1.0',
    author='Garrin Kimmell',
    author_email='garrin.kimmell@gmail.com',
    packages=['yieldmap'],
    scripts=[],
    url='http://pypi.python.org/pypi/YieldMap/',
    license='LICENSE.txt',
    description='Processing AgLeader Advanced Harvest Records.',
    long_description=open('README.md').read(),
    install_requires=[str(ir.req) for ir in install_reqs]
)
