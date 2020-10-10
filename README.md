
This repository contains the code to predict compressive and tensile strengths
of high-performance concrete as described in [this paper](add.link). 

If you find this code helpful, please cite 

```
@article{nguyen_efficient_2020,
    author = {Nguyen, Hoang and Vu, Thanh and Vo, Thuc P. and Thai, Huu-Tai},
		title={Efficient machine learning models for prediction of concrete strengths},
		journal={Construction and Building Materials},
    volume = {000},
    pages = {000--000},
		year={2020},
    doi = {---}
	}
```

### Requirements

- numpy 1.15.3
- scipy 1.1.0
- pandas 0.23.4
- scikit-learn 0.20.0
- xgboost 0.80


### How to run

```
  cd src
  python run_model.py
```

The problem (compressive/tensile concrete strengths) and model parameters
can be chosen in `src/run_model.py` 