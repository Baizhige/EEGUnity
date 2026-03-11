from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "readme.md"
VERSION_PATH = ROOT / "eegunity" / "_version.py"

version_ns = {}
exec(VERSION_PATH.read_text(encoding="utf-8"), version_ns)

setup(
    name="eegunity",
    version=version_ns["__version__"],
    packages=find_packages(include=["eegunity", "eegunity.*"]),
    install_requires=[
        "mne>=1.5.1",
        "numpy>=1.21.0",
        "matplotlib>=3.7.3",
        "h5py>=3.12.1",
        "openai>=1.35.1",
        "pdfplumber>=0.11.4",
        "pandas>=2.1.3",
        "scipy>=1.11.2",
        "scikit-learn>=1.3.2",
        "cloudpickle>=3.0.0",
        "wfdb>=4.1.2",
    ],
    author="EEGUnity Team",
    author_email="chengxuan.qin@outlook.com",
    description="An open source Python pacakge for large-scale EEG datasets processing",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Baizhige/EEGUnity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    package_data={
        "eegunity": ["resources/*.json"],
    },
)
