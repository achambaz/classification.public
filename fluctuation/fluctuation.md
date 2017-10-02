Fluctuation
================
Antoine Chambaz
26/9/2017

Les notions
-----------

-   Algorithme de Monte Carlo

-   Fluctuation

-   Interprétation algorithmique de l'entreprise d'apprentissage

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
  Y <- rnorm(QY(X), 1)
  dat <- cbind(X, Y)
  colnames(dat) <- c("X1", "X2", "Y")
  attr(dat, "QY") <- QY
  return(dat)
}
drawFromPzero(5)
```

    ##             X1         X2          Y
    ## [1,] 0.4290078  1.1093533  1.4611022
    ## [2,] 0.4984304 -0.8094265  0.9057240
    ## [3,] 0.1766923 -0.3274632  0.8792356
    ## [4,] 0.2743935  0.4566750 -0.4888658
    ## [5,] 0.2165102  0.3926771  0.3987509
    ## attr(,"QY")
    ## function (xx) 
    ## {
    ##     cos(2 * pi * xx[, 1]) + xx[, 2] + xx[, 2]^2/2
    ## }
    ## <bytecode: 0x41dfa58>
    ## <environment: 0x463d840>

-   Considérons par exemple la direction *s* définie ainsi:

``` r
s <- function(obs) {
}
```

[Retour à la table des matières](https://github.com/achambaz/laviemodedemploi#liens)
