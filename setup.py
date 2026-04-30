"""
Setup script for AI-Based Behavioral IDS package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-behavioral-ids-zerotrust",
    version="1.0.0",
    author="Research Team",
    description="AI-Based Behavioral Intrusion Detection System for Zero-Trust Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ai-behavioral-ids",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0"],
        "notebooks": ["jupyter>=1.0.0", "ipykernel>=6.25.0"],
    },
    entry_points={
        "console_scripts": [
            "ids-train=scripts.train_model:main",
            "ids-evaluate=scripts.evaluate_model:main",
            "ids-run=scripts.run_ids:main",
            "ids-simulate=simulation.run_simulation:main",
        ],
    },
)
