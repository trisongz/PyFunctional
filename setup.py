import sys
from setuptools import setup, find_packages

try:
    import pypandoc

    long_description = pypandoc.convert(
        "README.md", "rst", extra_args=["--columns=300"]
    )
except (IOError, ImportError):
    long_description = open("README.md").read()

common_install_requires = ["dill>=0.2.5", "tabulate<=1.0.0", "tensorflow>=2.3.0", "regularize", "dateparser", "pysimdjson"]
if "__pypy__" in sys.builtin_module_names:
    compression_requires = ["bz2file==0.98", "backports.lzma==0.0.6"]
    install_requires = common_install_requires
else:
    compression_requires = []
    install_requires = common_install_requires

setup(
    name="PyFunctional",
    description="Package for creating data pipelines with chain functional programming",
    long_description=long_description,
    url="https://github.com/trisongz/PyFunctional",
    author="Tri Songz - Forked from Pedro Rodriguez",
    author_email="ts@contentengine.ai",
    maintainer="Tri Songz",
    maintainer_email="ts@contentengine.ai",
    license="MIT",
    keywords="functional pipeline data collection chain rdd linq parallel",
    packages=find_packages(exclude=["contrib", "docs", "tests*", "test"]),
    version="1.4.3",
    install_requires=install_requires,
    extras_requires={
        "all": ["pandas"] + compression_requires,
        "compression": compression_requires,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
