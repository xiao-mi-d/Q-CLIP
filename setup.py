from setuptools import find_packages, setup


with open("requirements.txt", encoding="utf-8") as f:
    required = f.read().splitlines()


setup(
    name="qclip",
    version="0.1.0",
    description="Q-CLIP for video quality assessment.",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.11",
)
