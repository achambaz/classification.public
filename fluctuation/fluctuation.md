Fluctuation
================
Antoine Chambaz
2/10/2017

Les notions
-----------

-   Fluctuation

-   Interprétation algorithmique de l'entreprise d'apprentissage

-   Méthode de rejet

Fluctuation d'une loi
---------------------

``` r
set.seed(54321)
library(R.utils)
```

### La loi…

-   Considérons la loi *P*<sub>0</sub> suivante, qui produit des observations de la forme *O* = (*X*, *Y*)∈ℝ<sup>2</sup> × ℝ.

``` r
drawFromPzero <- function(nobs) {
  nobs <- Arguments$getInteger(nobs, c(1, Inf))
  X <- cbind(runif(nobs), rnorm(nobs))
  QY <- function(xx) {
    cos(2*pi*xx[, 1]) + xx[, 2] + xx[, 2]^2/2
  }
  Y <- runif(nobs, min = -1, max = 1) + QY(X)
  dat <- cbind(X, Y)
  colnames(dat) <- c("X1", "X2", "Y")
  attr(dat, "QY") <- QY
  return(dat)
}
five.obs <- drawFromPzero(5)
five.obs
```

    ##             X1         X2           Y
    ## [1,] 0.4290078  1.1093533  1.17780520
    ## [2,] 0.4984304 -0.8094265 -0.64470189
    ## [3,] 0.1766923 -0.3274632  0.09553343
    ## [4,] 0.2743935  0.4566750  0.55111090
    ## [5,] 0.2165102  0.3926771  0.58252583
    ## attr(,"QY")
    ## function (xx) 
    ## {
    ##     cos(2 * pi * xx[, 1]) + xx[, 2] + xx[, 2]^2/2
    ## }
    ## <bytecode: 0x3f6d600>
    ## <environment: 0x43dc1f0>

-   Considérons par exemple la direction *s* définie ainsi:

``` r
s <- function(obs) {
  QY <- attr(obs, "QY")
  if (is.null(QY) || !is.function(QY) || any(colnames(obs) != c("X1", "X2", "Y"))) {
    throw("Argument 'obs' should have the same  structure as that of an output
  of function 'drawFromPZero'")
  }
  clever.cov <- function(obs) {
    exp(obs[, "X1", drop = FALSE])
  }
  out <- ( obs[, "Y"] - QY(obs[, c("X1", "X2"), drop = FALSE]) ) * clever.cov(obs)
  out <- as.vector(out)
  attr(out, "clever.cov") <- clever.cov
  return(out)
}
s(five.obs)
```

    ## [1]  0.54560710  1.37796414 -0.08962619  0.18792555 -0.11935856
    ## attr(,"clever.cov")
    ## function (obs) 
    ## {
    ##     exp(obs[, "X1", drop = FALSE])
    ## }
    ## <bytecode: 0x3f26f50>
    ## <environment: 0x42c8b98>

-   La fonction *s* n'est pas bornée uniformément. Définissons la fluctuation «à travers le «noyau» *k*:

``` r
k <- function(tt) {
  2/(1+exp(-2*tt))
}

drawFromPeps <- function(nobs, direction, epsilon = 0) {
  nobs <- Arguments$getInteger(nobs, c(1, Inf))
  epsilon <- Arguments$getNumeric(epsilon)
  ##
  mode.dir <- mode(direction)
  if (!mode.dir == "function") {
    throw("Argument 'direction' should be a function, not", mode.dir)
  }
  two.obs <- drawFromPzero(2)
  test <- try(direction(two.obs))
  if (!is.null(attr(test, "class")) && attr(test, "class") == "try-error") {
    throw("Argument 'direction' is invalid:", test)
  }
  ##
  ## p_eps(x) = c(eps) k(eps s(x)) p_0(x)
  ##
  ## no need to compute c(eps)!
  ncurrent <- 0
  while(ncurrent < nobs) {
    U <- runif(nobs - ncurrent)
    obs.candidate <- drawFromPzero(nobs - ncurrent)
    keep <- ( U <= k(epsilon * direction(obs.candidate)) )
    if (any(keep)) {
      if (ncurrent == 0) {
        obs <- obs.candidate[keep, ]
      } else {
        obs <- rbind(obs, obs.candidate[keep, ])
      }
      ncurrent <- nrow(obs)
    }
  }
  attr(obs, "QY") <- function(){warning("Unknown\n")}
  return(obs)
}
five.obs.eps <- drawFromPeps(5, s, 0.1)
five.obs.eps
```

    ##             X1        X2          Y
    ## [1,] 0.1160398 1.1300149  1.7128531
    ## [2,] 0.8768612 1.3014683  3.7596315
    ## [3,] 0.3876749 0.4867082 -0.1149594
    ## [4,] 0.9096658 1.5777891  4.2001724
    ## [5,] 0.7496563 0.3123328  0.3366405
    ## attr(,"QY")
    ## function () 
    ## {
    ##     warning("Unknown\\n")
    ## }
    ## <bytecode: 0x422bf38>
    ## <environment: 0x46234a8>

-   L'approche ci-dessus est très générale. Notez qu'elle n'utilise la fonction `drawFromPzero` que pour *simuler* des données sous *P*<sub>0</sub>. Admettons maintenant que c'est l'espérance conditionnelle de *Y* sachant *X* sous *P*<sub>0</sub> (i.e., `QY`) que nous voulons en fait fluctuer. En exploitant le fait que (i) la fonction *s* est orthogonale à l'ensemble des fonctions de *X* seulement et (ii) que nous connaissons `QY`, nous pouvons construire une fluctuation plus spécifique et relative à la fonction de perte des moindres carrés (plutôt qu'à celle de l'opposée de la *l**o**g*-vraisemblance)…

``` r
drawFromPeps.bis <- function(nobs, direction, epsilon = 0) {
  nobs <- Arguments$getInteger(nobs, c(1, Inf))
  epsilon <- Arguments$getNumeric(epsilon)
  ##
  mode.dir <- mode(direction)
  if (!mode.dir == "function") {
    throw("Argument 'direction' should be a function, not", mode.dir)
  }
  two.obs <- drawFromPzero(2)
  test <- try(direction(two.obs))
  if (!is.null(attr(test, "class")) && attr(test, "class") == "try-error") {
    throw("Argument 'direction' is invalid:", test)
  }
  clever.cov <- attr(test, "clever.cov")
  ##
  obs <- drawFromPzero(nobs)
  QY <- attr(obs, "QY")
  Y <- rnorm(nobs, QY(obs) + epsilon * clever.cov(obs), sd = 1)
  attr(obs, "QY") <- direction
  return(obs)
}
five.obs.eps.bis <- drawFromPeps.bis(5, s, 0.1)
five.obs.eps.bis
```

    ##              X1          X2          Y
    ## [1,] 0.88977211 -1.11684127  0.7878125
    ## [2,] 0.37540584 -0.72177969 -1.8606739
    ## [3,] 0.57873104 -0.01226315 -1.6349697
    ## [4,] 0.01935932  0.11945715  0.3426051
    ## [5,] 0.56580960  0.52520244 -0.7839873
    ## attr(,"QY")
    ## function (obs) 
    ## {
    ##     QY <- attr(obs, "QY")
    ##     if (is.null(QY) || !is.function(QY) || any(colnames(obs) != 
    ##         c("X1", "X2", "Y"))) {
    ##         throw("Argument 'obs' should have the same  structure as that of an output\\n  of function 'drawFromPZero'")
    ##     }
    ##     clever.cov <- function(obs) {
    ##         exp(obs[, "X1", drop = FALSE])
    ##     }
    ##     out <- (obs[, "Y"] - QY(obs[, c("X1", "X2"), drop = FALSE])) * 
    ##         clever.cov(obs)
    ##     out <- as.vector(out)
    ##     attr(out, "clever.cov") <- clever.cov
    ##     return(out)
    ## }
    ## <bytecode: 0x3c61358>

[Retour à la table des matières](https://github.com/achambaz/laviemodedemploi#liens)
