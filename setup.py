import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfor",  # Replace with your own username
    version="0.0.1",
    author="Rod Ruoding Tian",
    author_email="ruodingt7@outlook.com",
    description="Operating room",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/trendiiau/tacore/src/master/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: N/A",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'yacs>=0.1.8',
    ],
)
