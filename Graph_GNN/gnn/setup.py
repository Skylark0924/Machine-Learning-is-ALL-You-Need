import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gnn",
    version="1.1.9",
    author="Matteo Tiezzi",
    author_email="mtiezzi@diism.unisi.it",
    description="Graph Neural Network Tensorflow implementation",
    long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/mtiezzi/gnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)