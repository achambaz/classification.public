Régression 'xgboost'
================
Antoine Chambaz
23/10/2017

Les notions
-----------

-   Algorithme de régression `xgboost`.

-   Ensembles d'apprentissage et de validation

-   'Machine-learning pipelines'

-   Reproductibilité

Fichier source
--------------

Afin d'extraire les portions de code `R` du fichier source [`regression.xgboost.Rmd`](https://github.com/achambaz/laviemodedemploi.develop/blob/master/regression.xgboost/regression.xgboost.Rmd), il suffit d'exécuter dans `R` la commande `knitr::purl("regression.xgboost.Rmd")`.

Préparation de la session `R`
-----------------------------

``` r
pkgs <- c("xgboost", "tidyverse", "devtools")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) {
    install.packages(pkg)
  }
}
pkg <- "pipelearner"
if (! (pkg %in% rownames(installed.packages()))) {
  devtools::install_github("drsimonj/pipelearner")
}
```

``` r
suppressMessages(library(tidyverse))
suppressMessages(library(lazyeval))
```

Une introduction à la régression par 'xgboost'
----------------------------------------------

``` r
set.seed(54321)
```

-   Mises à disposition par Météo France, ces données sont extraites du site [wikistat](https://github.com/wikistat). Nous souhaitons apprendre à prédire, à partir des données du jour, la concentration d'ozone le lendemain.

``` r
one_hot <- function(df) {
  as_tibble(stats::model.matrix(~.+0, data = df))
}

file <- file.path("http://www.math.univ-toulouse.fr/~besse/Wikistat/data", "depSeuil.dat")
ozone <- read_csv(file, col_names = TRUE) %>% one_hot %>% mutate_all(as.numeric)
```

    ## Parsed with column specification:
    ## cols(
    ##   JOUR = col_integer(),
    ##   O3obs = col_integer(),
    ##   MOCAGE = col_double(),
    ##   TEMPE = col_double(),
    ##   RMH2O = col_double(),
    ##   NO2 = col_double(),
    ##   NO = col_double(),
    ##   STATION = col_character(),
    ##   VentMOD = col_double(),
    ##   VentANG = col_double()
    ## )

``` r
## JOUR: jour férié (1) ou pas (0)
## O3obs: concentration d'ozone effectivement observée le lendemain à 17h locales (correspond souvent au maximum de pollution observée)
## MOCAGE: prévision de cette pollution obtenue par un modèle déterministe de mécanique des fluides
## TEMPE: température prévue par Météo France pour le lendemain 17h
## RMH2O: rapport d'humidité
## NO2: concentration en dioxyde d'azote
## NO: concentration en monoxyde d'azote
## STATION: lieu de l'observation (Aix-en-Provence, Rambouillet, Munchhausen, Cadarache et Plan de Cuques)
## VentMOD: force du vent
## VentANG: orientation du vent

head(ozone)
```

    ## # A tibble: 6 x 14
    ##    JOUR O3obs MOCAGE TEMPE   RMH2O   NO2    NO STATIONAix STATIONAls
    ##   <dbl> <dbl>  <dbl> <dbl>   <dbl> <dbl> <dbl>      <dbl>      <dbl>
    ## 1     1    91   93.2  21.5 0.00847 1.602 0.424          1          0
    ## 2     1   100  104.6  20.2 0.00881 2.121 0.531          1          0
    ## 3     0    82  103.6  17.4 0.00951 1.657 0.467          1          0
    ## 4     0    94   94.8  18.8 0.00855 2.350 0.701          1          0
    ## 5     0   107   99.0  23.7 0.00731 1.653 0.452          1          0
    ## 6     0   150  114.3  23.6 0.01182 5.316 1.343          1          0
    ## # ... with 5 more variables: STATIONCad <dbl>, STATIONPla <dbl>,
    ## #   STATIONRam <dbl>, VentMOD <dbl>, VentANG <dbl>

<!-- [lien intéressant](http://www.win-vector.com/blog/2017/04/encoding-categorical-variables-one-hot-and-beyond/)-->
Noter que nous avons dû transformer toutes les informations au format `numeric`. La variable `STATION` se prêterait volontiers à un recodage plus riche…

-   Préparation d'un ensemble d'apprentissage et d'un ensemble de validation.

``` r
m <- nrow(ozone)
val <- sample(1:m, size = round(m/3), replace = FALSE, prob = rep(1/m, m)) 
ozone.train <- ozone[-val, ]
ozone.test <- ozone[val, ]
```

-   Régression `xgboost`

Le nom `xgboost` est inspiré de l'expression «extreme gradient boosting» Cet algorithme d'apprentissage est aujourd'hui très apprécié dans la communauté du &laquo:machine learning». Vous trouverez une brève introduction aux principes sur lesquels cet algorithme est fondé [ici](http://xgboost.readthedocs.io/en/latest/model.html).

Sans finasser…

``` r
suppressMessages(library(xgboost))

get.rmse <- function(fit, newdata, target_var) {
  ## Convert 'newdata' object to data.frame
  newdata <- as.data.frame(newdata)
  # Get feature matrix and labels
  X <- newdata %>%
    select(-matches(target_var)) %>% 
    as.matrix()
  Y <- newdata[[target_var]]
  # Compute and return 'rmse'
  sqrt( mean((Y - predict(fit, X))^2) )
}


ozone.train.X <- select(ozone.train, -NO2) %>% as.matrix
ozone.train.Y <- ozone.train$NO2

nrounds <- 100
fit.xgboost.one <- xgboost(data = ozone.train.X, label = ozone.train.Y,
                           nrounds = nrounds, objective = "reg:linear", print_every = 10)
```

    ## [1]  train-rmse:3.794720 
    ## [11] train-rmse:0.433916 
    ## [21] train-rmse:0.184687 
    ## [31] train-rmse:0.137011 
    ## [41] train-rmse:0.101266 
    ## [51] train-rmse:0.065428 
    ## [61] train-rmse:0.053769 
    ## [71] train-rmse:0.042062 
    ## [81] train-rmse:0.027922 
    ## [91] train-rmse:0.021030 
    ## [100]    train-rmse:0.015879

``` r
rmse.test.one <- get.rmse(fit.xgboost.one, ozone.test, "NO2")
rmse.test.one
```

    ## [1] 0.9810754

Aurions-nous gagné à jouer sur les paramètres? Ici, nous passons la variable `eta` de sa valeur par défaut, `0.3`, à `0.1`.

``` r
params <- list(booster = "gbtree", objective = "reg:linear", eta = 0.1, gamma = 0,
               max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

fit.xgboost.two <- xgboost(data = ozone.train.X, label = ozone.train.Y,
                           nrounds = nrounds, params = params, print_every = 10)
```

    ## [1]  train-rmse:4.659818 
    ## [11] train-rmse:2.007874 
    ## [21] train-rmse:0.980167 
    ## [31] train-rmse:0.539510 
    ## [41] train-rmse:0.332184 
    ## [51] train-rmse:0.233994 
    ## [61] train-rmse:0.183641 
    ## [71] train-rmse:0.154095 
    ## [81] train-rmse:0.134877 
    ## [91] train-rmse:0.122614 
    ## [100]    train-rmse:0.116154

``` r
rmse.test.two <- get.rmse(fit.xgboost.two, ozone.test, "NO2")
rmse.test.two
```

    ## [1] 1.02734

Ou bien, aurions-nous gagné à stopper les itérations plus tôt?

``` r
params <- list(booster = "gbtree", objective = "reg:linear", eta = 0.3, gamma = 0,
               max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

fit.xgboost.cv <- xgb.cv(data = ozone.train.X, label = ozone.train.Y,
                         nrounds = nrounds, nfold = 5, params = params, print_every = 10,
                         showsd = TRUE, early.stopping.rounds = 20, maximize = FALSE)
```

    ## [1]  train-rmse:3.799698+0.136626    test-rmse:3.821138+0.626400 
    ## [11] train-rmse:0.431261+0.046820    test-rmse:1.513854+0.652663 
    ## [21] train-rmse:0.165710+0.014164    test-rmse:1.562004+0.659993 
    ## [31] train-rmse:0.109532+0.009998    test-rmse:1.569076+0.664805 
    ## [41] train-rmse:0.073094+0.006837    test-rmse:1.570267+0.664585 
    ## [51] train-rmse:0.048866+0.002317    test-rmse:1.571713+0.663063 
    ## [61] train-rmse:0.033369+0.003329    test-rmse:1.572273+0.662416 
    ## [71] train-rmse:0.023395+0.002403    test-rmse:1.572423+0.662509 
    ## [81] train-rmse:0.017311+0.001902    test-rmse:1.572235+0.662476 
    ## [91] train-rmse:0.012581+0.001232    test-rmse:1.572238+0.662429 
    ## [100]    train-rmse:0.009421+0.001441    test-rmse:1.572217+0.662248

``` r
best.xgboost.count <- which.min(fit.xgboost.cv$evaluation_log$test_rmse_mean)
fit.xgboost.three <- xgboost(data = ozone.train.X, label = ozone.train.Y,
                             nrounds = best.xgboost.count, params = params, print_every = 10)
```

    ## [1]  train-rmse:3.794720 
    ## [11] train-rmse:0.433916

``` r
rmse.test.three <- get.rmse(fit.xgboost.three, ozone.test, "NO2")
rmse.test.three
```

    ## [1] 1.004428

-   Mise en place d'une «ML pipeline»

Cette section s'inspire de [ce billet](https://drsimonj.svbtle.com/with-our-powers-combined-xgboost-and-pipelearner)…

``` r
pl.xgboost <- function(data, formula, ...) {
  data <- as.data.frame(data)

  X_names <- as.character(f_rhs(formula))
  y_name  <- as.character(f_lhs(formula))

  if (X_names == '.') {
    X_names <- names(data)[names(data) != y_name]
  }

  X <- data.matrix(data[, X_names])
  y <- data[[y_name]]

  xgboost(data = X, label = y, ...)
}

fit.xgboost.four <- pl.xgboost(ozone.train, NO2 ~ ., nrounds = nrounds, objective = "reg:linear", verbose = 0)

rmse.test.four <- get.rmse(fit.xgboost.four, ozone.test, "NO2")
rmse.test.four
```

    ## [1] 0.9810754

-   Nous voilà enfin prêts à procéder à l'évaluation d'une variété de paramétrisations de `xgboost` par «ML pipelining»

``` r
suppressMessages(library(pipelearner))

pl <- pipelearner(ozone.train, pl.xgboost, NO2 ~ .,
                  nrounds = seq(10, 25, 5),
                  eta = c(.1, .3, .5),
                  gamma = c(0, 0.1, 0.2),
                  max_depth = c(4, 6),
                  verbose = 0)
```

    ## Warning: `cross_d()` is deprecated; please use `cross_df()` instead.

    ## Warning: `cross_d()` is deprecated; please use `cross_df()` instead.

``` r
fits.xgboost <- pl %>% learn()

results <- fits.xgboost %>% 
  mutate(
    ## hyperparameters
    nrounds   = map_dbl(params, "nrounds"),
    eta       = map_dbl(params, "eta"),
    max_depth = map_dbl(params, "max_depth"),
    ## rmse
    rmse.train = pmap_dbl(list(fit, train, target), get.rmse),
    rmse.test  = pmap_dbl(list(fit, test,  target), get.rmse)
  ) %>% 
  ## Select columns and order rows
  select(nrounds, eta, max_depth, contains("rmse")) %>% 
  arrange(desc(rmse.test), desc(rmse.train))

results
```

    ## # A tibble: 72 x 5
    ##    nrounds   eta max_depth rmse.train rmse.test
    ##      <dbl> <dbl>     <dbl>      <dbl>     <dbl>
    ##  1      10   0.1         6   2.054802  3.204319
    ##  2      10   0.1         4   2.082638  3.191337
    ##  3      10   0.1         4   2.082638  3.191337
    ##  4      10   0.1         4   2.082638  3.191337
    ##  5      10   0.1         6   2.057249  3.178219
    ##  6      10   0.1         6   2.056468  3.170759
    ##  7      15   0.1         4   1.416361  2.785399
    ##  8      15   0.1         4   1.416361  2.785399
    ##  9      15   0.1         4   1.416407  2.784853
    ## 10      15   0.1         6   1.373803  2.768126
    ## # ... with 62 more rows

results &lt;- fits %&gt;% mutate( \# hyperparameters nrounds = map\_dbl(params, "nrounds"), eta = map\_dbl(params, "eta"), max\_depth = map\_dbl(params, "max\_depth"), \# Accuracy accuracy\_train = pmap\_dbl(list(fit, train, target), accuracy), accuracy\_test = pmap\_dbl(list(fit, test, target), accuracy) ) %&gt;% \# Select columns and order rows select(nrounds, eta, max\_depth, contains("accuracy")) %&gt;% arrange(desc(accuracy\_test), desc(accuracy\_train))

results

[Retour à la table des matières](https://github.com/achambaz/laviemodedemploi.develop#liens)
