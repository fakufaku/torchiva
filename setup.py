import pathlib

from setuptools import find_packages, setup

# obtain long description from the readme
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.rst").read_text(encoding="utf-8")

setup(
    name="torchiva",
    version="0.1.1",
    description="Package for independent vector analysis in torch",
    long_description=long_description,
    long_description_content_type="text/x-rst",  # text/plain, text/x-rst, text/markdown
    url="https://github.com/fakufaku/torchiva",
    author="Robin Scheibler",
    # author_email='author@example.com',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # keywords='sample, setuptools, development',
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "torchaudio",
        "PyYAML",
    ],
    extras_require={
        "all": [],
    },
)
