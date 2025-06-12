<!-- <div align="center"> -->
<img src="https://github.com/OgawaSama/air-quality/raw/tree.png" alt="tiny tree" width="500"/>

<!-- </div> -->

# Decision Trees for Air Quality Dataset

This program creates a decision tree for solving the Air Quality assessment problem, where multiple variables may help determine a region's air quality.
The `data.csv` is from [kaggle](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) by Mujtaba Mateen.

This program allows the user to specify a method in which to determine the Tree's depth, their targetted accuracy, which rendering engine to use and whether or not to run benchmarks instead.

The default parameters are:
* No maximum depth defined; runs until exhausted
* If `--depth minimum` is selected, target accuracy is 0.85
* Test/Train ratio is 30/70
* `data.csv` as the dataset file
* `tree.png` as the Tree's image output file
* `metrics.csv` as the benchmark's output file


## Running the program

### Windows
To run this program, simply create and activate a virtual environment with
```shell
python -m venv [envname]
[envname]\Scripts\activate
```

Install the needed packages with
```shell
pip install -r requirements.txt
```

And then run the program with
```shell
python decisiontree.py
```
### Linux
To run this program, simply create and activate a virtual environment with
```shell
python -m venv [envname]
source [envname]/bin/activate
```

Install the needed packages with
```shell
pip install -r requirements.txt
```

And then run the program with
```shell
python decisiontree.py
```

