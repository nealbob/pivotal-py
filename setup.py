import os
from setuptools import setup, find_packages

setup(
    name="pivotal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lark",
        "pandas",
        "matplotlib",
        "great-tables",
        "sqlalchemy",
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    author="User",
    description="A DSL parser for pivotal-py",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
)
