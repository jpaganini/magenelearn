from setuptools import setup, find_packages

setup(
    name="maGeneLearn",
    version="0.1.2",
    description="A CLI wrapper for the maGeneLean ML pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jpaganini/magenelearn",              # ← homepage link
    project_urls={                                               # ← additional links
        "Documentation": "https://github.com/jpaganini/magenelearn#readme",
        "Source": "https://github.com/jpaganini/magenelearn",
        "Tracker": "https://github.com/jpaganini/magenelearn/issues",
    },
    author="Julian A. Paganini",
    author_email="j.a.paganini@uu.nl",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires='>=3.9',
    install_requires=[
            "click==8.1.7",
            "pandas==2.1.1",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "imbalanced-learn==0.11.0",
            "xgboost==2.0.3",
            "joblib==1.2.0",
            "shap==0.42.1",
            "matplotlib==3.7.2",
	    "py-muvr==1.0.1"
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
    include_package_data=True,
)
