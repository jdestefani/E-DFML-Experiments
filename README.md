# E-DFML-Experiments

Repository containing the code for the experiments in the paper "Factor-based framework for multivariate and multi-step ahead forecasting of large scale time series"

## Requirements
```R
# Availble via CRAN
install.packages(c("tensorflow","keras","forecast","dplyr","abind",
                   "vars","DMwR","lightgbm")

devtools::install_github("gbonte/gbcode")
devtools::install_github("jdestefani/MultivariateBenchmarksTS") #Soon available
devtools::install_github("jdestefani/ExtendedDFML") #Soon available
```

## Quickstart
```
# For Electricity experiment
Rscript ExtendedDFMLArticle.R 1

# For Mobility experiment
Rscript ExtendedDFMLArticle.R 2

# For Traffic experiment
Rscript ExtendedDFMLArticle.R 3
```
