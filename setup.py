# -*- coding: utf-8 -*-
from pathlib import Path
import json

from setuptools import setup, find_packages

with open('./ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']

setup(
    name='ocrd_trcor',
    version=version,
    description='OCR-D processor for the TrOCR engine',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Mike Gerber',
    author_email='mike.gerber@sbb.spk-berlin.de',
    url='https://github.com/qurator-spk/ocrd_trocr',
    license='Apache License 2.0',
    packages=find_packages(exclude=('test', 'docs')),
    install_requires=Path('requirements.txt').read_text().split('\n'),
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-trocr-recognize=ocrd_trocr.cli:ocrd_trocr_recognize',
        ]
    },
)
