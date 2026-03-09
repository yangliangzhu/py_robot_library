from setuptools import setup, find_packages

setup(
    name="py_library",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for robot models and tools",
    packages=find_packages(),
    package_data={
        "model": ["configs/*.yaml"],
    },
    install_requires=[
        "numpy",
        "casadi",
        "pyyaml",
    ],
    python_requires=">=3.6",
)
