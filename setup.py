from setuptools import setup, find_packages

setup(
    name="sciforge",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pytest>=6.0.0",
        "sphinx>=4.0.0",
        "black>=21.0"
    ],
    python_requires=">=3.7",
) 