from setuptools import setup, find_packages

import versioneer

setup(
    name='canopyrs',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    requires=[],
    install_requires=[
        # list your project's dependencies here
        # e.g., 'numpy', 'pandas'
    ],
)
