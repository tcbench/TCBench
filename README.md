# TCBench - Alpha Repository
 
Welcome to TCBench, our platform and benchmark dataset for data-driven tropical cyclone studies.

Graphical Summary of TCBench:
![TCBench Graphical Summary](https://github.com/tcbench/TCBench/blob/c3452b3282b405ee41d062eba0b89051987780d0/Figure_Summary.png?raw=True)

## Background

Coastal risks and vulnerability to tropical cyclone (TC)-driven heavy rainfall, flooding, and storm surge are increasing due to slower, wetter TCs; poleward expansion of maximum potential TC intensity; and growing population near coasts. However, predicting TC intensity variations remains a challenge even for major global storm forecasting centers due to complex storm dynamics. In particular, the prediction of rapid intensification remains especially challenging. TC intensity has been linked with TC rain rates, but the scaling and spatial variability of this relationship is still being studied. Still, as the climate warms both wind speeds and rainfall rates are projected to increase and thus a strong foundational dataset for tropical cyclone intensity and precipitation predictions is crucial for future risk management and coastal resilience.

We note that the World Meteorological Organization has, as part of its TC Programme, designated ten Regional Specialized Meteorological Centers for operational TC forecasting. While these incorporate satellites, statistical and numerical models to monitor and forecast cyclogenesis and intensification, it is currently difficult to compare ML models that aim to improve TC predictions to the appropriate observations and previous predictions. This is a gap we hope to fill by fusing the archives from several RSMCs into a single data repository.

Machine Learning (ML) models are widely used to improve TC genesis, track, and intensity predictions. However, even though some models are able to outperform NHC operational prediction of RI over the Atlantic and East Pacific ocean basins, most ML groups use standard input lists (e.g., those provided by the NHC). As such, which predictors are most predictive of TC intensity changes remains an open question. Furthermore, an overarching issue in studies applying modern ML to TC intensity prediction is that these studies usually define different targets and evaluation metrics, preventing an objective comparison of different frameworks. This further motivates grouping standard tropical meteorology definitions and evaluation protocols in a unified data repository, e.g., by creating a TC benchmark dataset for ML applications.

The use of benchmark datasets is further motivated by the fact that several scientific domains have shown that defining a benchmark dataset helps identify optimal ML solutions for a specific challenge by enabling cross-institutional collaborations and healthy competition. In atmospheric science, WeatherBench, which frames weather forecasting from global reanalysis as a statistical regression problem, has encouraged scientific reflection on atmospheric dynamics, scale interaction, predictability, and uncertainty that goes beyond the eight entries on its leaderboard. More recent attempts use sophisticated ML methods, such as graph neural networks, adaptive Fourier neural operators, and diffusion-based generators, confirming the innovation potential of such benchmark datasets. However, even though ML benchmark datasets for weather and climate applications are progressively appearing in response to community enthusiasm, few focus on extreme events despite their critical importance for operations.

## Where TCBench comes in

Our goal, then, is to provide open, user-friendly data processing tools, evaluation protocols, visualization tools, and baseline prediction models to benefit the atmospheric science and AI communities. By facilitating a unified evaluation of ML models of tropical cyclones, we hope to give the scientific community a clear path towards developing skillful data-driven predictions of tropical cyclones for both present and future climates.

TCBench aims to provide opportunities to study the predictability of tropical cyclones (and changes in behavior associated with changing climate), as well as developing a dataset and evaluation tools that can be used freely by the scientific community.

## Getting Started

Please refer to our getting started jupyter notebook: Getting_Started.ipynb. Special thanks to Samuel Darmon for his work on it.

### Folder Structure
```
tcbench/
├── .gitignore
├── __init__.py
├── Figure_Summary.png
├── LICENSE
├── README.md
├── requirements.txt
└── dev/
    ├── __init__.py
    ├── baselines.py
    ├── climatology_maker.py
    ├── compute_clim_results.py
    ├── compute_persistence.py
    ├── data_preprocessing.py
    ├── eval_plotter.py
    ├── evaluate_tracks.py
    ├── evaluate_tracks_RI.py
    ├── Getting_Started.ipynb
    ├── metrics.py
    ├── metrics_test.py
    ├── plot_errors.py
    ├── plot_RI.py
    ├── postprocessing_model_dfBuilder.py
    ├── postprocessing training (deep).py
    ├── postprocessing training (linear).py
    ├── TempestExtremes_example.sh
    ├── track_matcher.py
    └── utils/
        ├── __init__.py
        ├── constants.py
        ├── data_lib.py
        ├── ML_functions.py
        └── toolbox.py

```
`\dev` contains all of the python scripts needed to evaluate your tracks. <br>
`\dev\utils\constants.py` holds reference values for some operations carried out by TCBench. This includes classes describing the tracks provided by, e.g., IBTrACS, which facilitate track processing.  <br>
`\dev\utils\toolbox.py` includes support functions and classes used during evaluation.
`\dev\utils\ML_functions.py` includes support functions/classes for the post-processing models.
`\dev\utils\data_lib.py` includes support functions/classes for the post-processing models.

