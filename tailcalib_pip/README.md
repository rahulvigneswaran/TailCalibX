# tailcalib

__tailcalib__ is a Python library for balancing a __long-tailed / imbalanced__ dataset by generating synthetic datapoints which will inturn increase the class-wise and overall test accuracy on the original dataset. 


This package is based on the paper [Feature Generation for Long-tail Classification](https://github.com/rahulvigneswaran/TailCalibX) by [Rahul Vigneswaran](https://rahulvigneswaran.github.io/), [Marc T. Law](http://www.cs.toronto.edu/~law/), [Vineeth N. Balasubramanian](https://lab1055.github.io/), [Makarand Tapaswi](https://makarandtapaswi.github.io/).

For much more detailed experiments, code and instructions, check [rahulvigneswaran/TailCalibX](https://github.com/rahulvigneswaran/TailCalibX) [![Star on GitHub](https://img.shields.io/github/stars/rahulvigneswaran/TailCalibX.svg?style=social)](https://github.com/rahulvigneswaran/TailCalibX/stargazers)
.
## 💻 Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install __tailcalib__.

```bash
pip install tailcalib
```

## 👨‍💻 Basic Usage

```python
# Import
from tailcalib import tailcalib

# Initialize
a = tailcalib(base_engine="numpy")   # Options: "numpy", "pytorch"

# Imbalanced random fake data
import numpy as np
X = np.random.rand(200,100)
y = np.random.randint(0,10, (200,))

# Balancing the data using "tailcalib"
feat, lab, gen = a.generate(X=X, y=y)

# Output comparison
print(f"Before: {np.unique(y, return_counts=True)}")
print(f"After: {np.unique(lab, return_counts=True)}")
```

## 🧪 Advanced Usage
### 🧩 Sample code

```python
# Import
from tailcalib import tailcalib

# Initialize
a = tailcalib(base_engine="numpy")   # Options: "numpy", "pytorch"

# Imbalanced random fake data
import numpy as np
# Train data
X_train = np.random.rand(200,100)
y_train = np.random.randint(0,10, (200,))
# Test data
X_test = np.random.rand(20,100)
y_test = np.random.randint(0,10, (20,))

# Balancing the data using "tailcalib". 
# Try to play with the other hyperparameters to get a better generated datapoint.
feat, lab, gen = a.generate(X=X_train, y=y_train, tukey_value=1.0, alpha=0.0, topk=1, extra_points=0, shuffle=True)

# Always remember to convert the val/test data before doing validation/testing.
X_test, y_test = a.convert_others(X=X_test, y=y_test)

# Output comparison
print(f"Before: {np.unique(y_train, return_counts=True)}")
print(f"After: {np.unique(lab, return_counts=True)}")
```

### ⚙ Arguments
- `X` : Features
- `y` : Corresponding labels
- `tukey_value` : Value to convert any distrubution of data into a normal distribution. Defaults to 1.0.
- `alpha` : Decides how spread out the generated data is. Defaults to 0.0.
- `topk` : Decides how many nearby classes should be taken into consideration for the mean and std of the newly generated data. Defaults to 1.
- `extra_points` : By default the number of datapoints to be generated is decided based on the class with the maximum datapoints. This variable decides how many more extra datapoints should be generated on top of that. Defaults to 0.
- `shuffle` : Shuffles the generated and original datapoints together. Defaults to True.

### 📤 Returns:
- `feat_all` : Tukey transformed train data + generated datapoints
- `labs_all` : Corresponding labels to feat_all
- `generated_points` : Dict that consists of just the generated points with class label as keys.
        


## 🪀 Results on a Toy Dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Yj2qymSm3NgCBqvKn5r_cOiEFl9wGp3J?usp=sharing)

The higher the `Imb ratio`, the more imbalanced the dataset is.
`Imb ratio = maximum_sample_count/minimum_sample_count`.

Check [this notebook](https://colab.research.google.com/drive/1Yj2qymSm3NgCBqvKn5r_cOiEFl9wGp3J?usp=sharing) to play with the toy example from which the plot below was generated.
![](toy_example_output.svg)

## 📃 Citation
If you use this package in any of your work, cite as,
```
@inproceedings{rahul2021tailcalibX,
    title   = {{Feature Generation for Long-tail Classification}},
    author  = {Rahul Vigneswaran, Marc T. Law, Vineeth N. Balasubramanian, Makarand Tapaswi},
    booktitle = {ICVGIP},
    year = {2021}
}
```

## 👁 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## ❤ About me
[Rahul Vigneswaran](https://rahulvigneswaran.github.io/)

## ✨ Extras
[🐝  Long-tail buzz](https://rahulvigneswaran.github.io/longtail-buzz/) : If you are interested in deep learning research which involves __long-tailed / imbalanced__ dataset, take a look at [Long-tail buzz](https://rahulvigneswaran.github.io/longtail-buzz/) to learn about the recent trending papers in this field.


<!-- <iframe
  src="https://rahulvigneswaran.github.io/longtail-buzz/"
  style="width:100%; height:500px;"
></iframe> -->

## 📝 License
[MIT](LICENSE)
