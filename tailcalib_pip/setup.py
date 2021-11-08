import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tailcalib",                    
    version="0.0.1",                       
    author="Rahul Vigneswaran",                  
    description="tailcalib is a Python library for balancing a long-tailed / imbalanced dataset by generating synthetic datapoints which will inturn increase the class-wise and overall test accuracy on the original dataset.",
    long_description=long_description,     
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),   
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      
    python_requires='>=3.6',                
    py_modules=["tailcalib"],             
    package_dir={'':'src'},    
    install_requires=["numpy", "scikit-learn"]                     
)