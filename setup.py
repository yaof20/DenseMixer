from setuptools import setup, find_packages

setup(
    name="densemixer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.51.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.12",
    entry_points={
        'console_scripts': [
            'densemixer = densemixer.commands:run',
        ],
    },
)