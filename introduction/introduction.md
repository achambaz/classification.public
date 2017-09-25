Introduction
================
Antoine Chambaz
25/9/2017

Une introduction à la classification selon les plus proches voisins
-------------------------------------------------------------------

``` r
set.seed(54321) ## reproductibilité...
```

### Les données «iris»

-   Préparatifs:

``` r
suppressMessages(library(caret))
data(iris)
species.col <- grep("Species", colnames(iris))
m <- nrow(iris)
val <- sample(1:m, size = round(m/3), replace = FALSE, prob = rep(1/m, m)) 
iris.train <- iris[-val, ]
iris.test <- iris[val, ]
```

-   Pour obtenir une description du jeu de données, exécuter `?iris`.

``` r
nb.neighbors <- 4
trained.knn <- knn3(Species ~ ., iris.train, k = nb.neighbors)
test.probs.knn <- predict(trained.knn, iris.test)
test.preds.knn <- colnames(test.probs.knn)[apply(test.probs.knn, 1, which.max)]
perf.knn <- table(test.preds.knn, iris.test[, species.col])
perf.knn
```

    ##               
    ## test.preds.knn setosa versicolor virginica
    ##     setosa         16          0         0
    ##     versicolor      0         13         1
    ##     virginica       0          1        19

-   Pour obtenir une description de la fonction `knn3`, exécuter `?knn3` (généralisation évidente).

``` r
suppressMessages(library(kknn))
## issu de l'aide de la fonction 'kknn'
trained.kknn <- kknn(Species~., iris.train, iris.test, distance = 1,
    kernel = "triangular")
```

-   Pour accéder à un résumé de l'objet `trained.knn`, exécuter `summary(trained.kknn)`.

``` r
test.preds.kknn <- fitted(trained.kknn)
perf.kknn <- table(iris.test$Species, test.preds.kknn)
perf.kknn
```

    ##             test.preds.kknn
    ##              setosa versicolor virginica
    ##   setosa         16          0         0
    ##   versicolor      0         11         3
    ##   virginica       0          1        19

-   Visualisation des résultats.

``` r
pcol <- as.character(as.numeric(iris.test$Species))
## knn3
pairs(iris.test[1:4], pch = pcol, col = c("green3", "red")
[(iris.test$Species != test.preds.kknn)+1])
```

![](img/visualisation-1.png)

``` r
## kknn
pairs(iris.test[1:4], pch = pcol, col = c("green3", "red")
[(iris.test$Species != test.preds.knn)+1])
```

![](img/visualisation-2.png)
