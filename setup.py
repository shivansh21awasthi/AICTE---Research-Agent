"""
setup.py — Package setup for Research Agent.
"""

from setuptools import setup, find_packages

setup(
    name="research-agent",
    version="1.0.0",
    description="AI Research Agent powered by IBM Granite on IBM watsonx.ai",
    author="AICTE Research Agent Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "output*"]),
    install_requires=[
        "ibm-watsonx-ai>=1.1.2",
        "ibm-cloud-sdk-core>=3.20.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "arxiv>=2.1.0",
        "beautifulsoup4>=4.12.3",
        "reportlab>=4.2.0",
        "pandas>=2.2.0",
        "colorama>=0.4.6",
        "rich>=13.7.1",
        "click>=8.1.7",
    ],
    extras_require={
        "discovery": ["ibm-watson>=7.0.1"],
        "pdf-extraction": ["pdfminer.six>=20221105", "pypdf2>=3.0.1"],
        "scholarly": ["scholarly>=1.7.11"],
    },
    entry_points={
        "console_scripts": [
            "research-agent=main:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
