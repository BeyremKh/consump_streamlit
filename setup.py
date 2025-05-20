
from setuptools import setup, find_packages
setup(
    name="consumption-app",
    version="0.1.0",
    packages=find_packages(include=['self_consumption*']),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.22.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "plotly>=5.3.0",
        "matplotlib>=3.4.0",
        "demandlib>=0.2.2"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "isort>=5.0.0",
            "mypy>=0.812",
            "flake8>=3.9.0"
        ]
    },
    author="Beyrem Khadhraoui",
    author_email="beyremkh@gmail.com",
    description="Streamlit app for energy consumption analysis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BeyremKh/consump_streamlit",
    project_urls={
        "Bug Tracker": "https://github.com/BeyremKh/consump_streamlit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
