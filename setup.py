from setuptools import setup, find_packages

setup(
    name="maGeneLearn",
    version="0.1.0",
    description="A CLI wrapper for the maGeneLean ML pipeline",
    author="Julian A. Paganini",
    author_email="j.a.paganini@uu.nl",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "click>=7.0",
        # add other runtime dependencies here
    ],
    entry_points={
        "console_scripts": [
            "maGeneLearn = maGeneLearn.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
