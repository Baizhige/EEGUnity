from setuptools import setup, find_packages

setup(
    name='eegunity',
    version='0.5.5',
    packages=find_packages(include=['eegunity', 'eegunity.*']),
    install_requires=[
        'mne>=1.5.1',
        'numpy>=1.21.0,<2.0.0', # Required due to compatibility limits of NumPy and MNE with Python 3.13
        'matplotlib>=3.7.3',
        'h5py>=3.12.1',
        'openai==1.35.1',
        'pdfplumber>=0.11.4',
        'pandas>=2.1.3',
        'scipy>=1.11.2'
    ],
    author='EEGUnity Team',
    author_email='chengxuan.qin@outlook.com',
    description='An open source Python pacakge for large-scale EEG datasets processing',
    long_description=open('readme.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Baizhige/EEGUnity',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
],
    python_requires='>=3.7, <3.13', # Required due to compatibility limits of NumPy and MNE with Python 3.13
    package_data={
            'eegunity': ['resources/*.json'],
        },
)
