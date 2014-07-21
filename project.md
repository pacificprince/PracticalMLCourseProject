---
title:  'Predicting Quality of Personal Activity using Machine Learning Algorithms'
author: Student
date: July 19, 2011
---
\  



## Summary
****
Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is
now possible to collect a large amount of data about personal activity
relatively inexpensively. In this project, the
goal is to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants to quantify *how well* a particular activity is done. 
The participants were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

## Data Loading 
****
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 
First the data is downloaded. On inspection it is found that
there is missing data denoted by `NA` or `#DIV/0!`. These are replaced by `NA`.


```r
DownloadAndReadFile <- function(file.url, filename, nastrings){
  # Download training data if not already download
  if (!file.exists(filename)){
    download.file(file.url, destfile = filename)
    download.date <- date()  # Date stamp the downloaded data
  }
  
  read.csv(filename, header = TRUE, na.strings = nastrings)  
}

training.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing.url  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

nastrings <- c("NA", "#DIV/0!")
data.training <- DownloadAndReadFile(training.url, "training.csv", nastrings)
data.testing  <- DownloadAndReadFile(testing.url,  "testing.csv",  nastrings)
```

The training data contains 19622 rows and 160 columns.

```r
dim(data.training)
```

```
[1] 19622   160
```

And the testing data contains 20 rows and 160 columns

```r
dim(data.testing)
```

```
[1]  20 160
```


## Data Pre-processing
****
As mentioned above, the *training* and *testing* data sets contain missing 
values. So columns with `NA` are removed in the datasets. Also, columns that
are not relevant to any sensor measurements are removed since these cannot be 
used to build a model.

### Feature Selection

```r
na.cols       <- colSums(!is.na(data.training)) == nrow(data.training)
data.training <- data.training[, na.cols]; 
data.testing  <- data.testing[, na.cols];  

rm.cols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",  
             "cvtd_timestamp", "new_window",  "num_window")
data.training <- data.training[, !(colnames(data.training) %in% rm.cols)]
data.testing  <- data.testing[, !(colnames(data.testing) %in% rm.cols)]
```

The *training* and *testing* sets now have 53 columns. All columns 
in the training set are `numeric` or `int` except for `classe` (variable of interest) 
which is a `factor` (as desired). 

```r
str(data.training)
```

```
'data.frame':	19622 obs. of  53 variables:
 $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
 $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
 $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
 $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
 $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
 $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
 $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
 $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
 $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
 $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
 $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
 $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
 $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
 $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
 $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
 $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
 $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
 $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
 $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
 $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
 $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
 $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
 $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
 $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
 $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
 $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
 $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
 $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
 $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
 $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
 $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
 $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
 $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
 $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
 $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
 $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
 $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
 $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
 $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
 $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
 $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
 $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
 $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
 $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
 $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
 $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
 $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
 $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
 $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
 $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
 $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
 $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
 $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

### Training and Validation Sets 
The original *training* set is further subdivided into *training* and *validation* sets
for the purposes of modeling. The random number generator (RNG) state is (seed) is set to 
12321. The *training* set is partitioned so as to contain 70% of the original data
(13737 rows). The remainder of the data (5885 rows) is assigned to the `validation` set.

```r
set.seed(12321)

library(caret)
```

```
Loading required package: lattice
Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y = data.training$classe, p = 0.70, list = FALSE)

training   <- data.training[inTrain, ] 
validation <- data.training[-inTrain, ]

testing <- data.testing[, -ncol(data.testing)]
```

## Predictive Modeling
****
> Since the *classe* variable is a factor this is a **classification** problem.  

The strategy is to use the following four machine learning methods:

* Random Forest 
* Bagged CART
* Support Vector Machines with Radial Basis Function Kernel 
* k-Nearest Neighbors

Followed by an *ensemble* model. 

> Throughout this project, the *caret* package is used for training, 
> validation and testing the models.

A `models` object is used to store all information. The object contains the 
following parameters:  

--------------------------------------------------------------
Parameter                  Information
-------------------------- -----------------------------------
name                        List of model names

method                      Character strings corresponding to each
                            model in models\$name given as argument
                            to caret's `train` function 
            
fit                         Fit information returned from `train`      
                            function corresponding to each model in
                            models\$name 
            
pred\$validation            Predictions of models on validation set  

pred\$testing	              Predictions of models on testing set     

confmat	                    Confusion matrix for each model applied 
                            to the validation set                 
                            
confmat\$summary	          Dataframe of accuracy parameters from each model  

ensemble\$fit	              Fit information for *ensemble* model using the 
                            Random Forest algorithm        
                            
ensemble\$pred\$validation	Predictions of *ensemble* model on validation set 

ensemble\$pred\$testing	    Predictions of *ensemble* model on testing set   

ensemble\$confmat	          Confusion matrix for *ensemble* model 
                            applied to the validation set               
                
--------------------------------------------------------------

### FitModel Function
We begin by defining a generic function which contains training options common
to all models in the `train` function. This function takes the training data,
`data.train`, and the model method, `model.method` and applies the `train` function 
to create the model.

A *repeated cross-validation* scheme is used with 
*4 folds and 1 repeat*. PCA is used as a pre-processing algorithm and 
verbosity is turned off.

```r
FitModel <- function(data.train, model.method = "rf"){
  
  trn.ctrl <- trainControl(method = 'repeatedcv', 
                           number = 4,            repeats = 1, 
                           returnResamp = 'none', classProbs = TRUE,
                           returnData = FALSE,    savePredictions = TRUE, 
                           verboseIter = FALSE,   allowParallel=TRUE)  
  
  model <- train(classe ~ ., data = data.train, method = model.method,
                 trControl = trn.ctrl, preProcess = c("pca"), 
                 verbose = FALSE, trace = FALSE)
  
  list(model)
  
}
```

### Train Models
With the FiModel function above, it is now simple to obtain a list of models to
the training data using `sapply` as shown below. All fit information is stored in 
the `models` object. Note that the RNG seed is set to 97611.


```r
library(randomForest)
```

```
randomForest 4.6-7
Type rfNews() to see new features/changes/bug fixes.
```

```r
library(ipred)
library(kernlab)
library(kknn)
```

```

Attaching package: 'kknn'

The following object is masked from 'package:caret':

    contr.dummy
```

```r
set.seed(97611)

models <- c()

models$name   <- c("RandomForest", "BaggedCART", "SVMRadial", "kNearestNeighbors")
models$method <- c("rf", "treebag", "svmRadial", "kknn")

models$fit <- sapply(models$method, 
                     function(x) FitModel(training, model.method = x))
```

```
Loading required package: plyr
```

```r
nmodels <- length(models$fit)
```

#### Predictions on Validation Set (Out-of-Sample)
Predictions for each model on the validation data set are obtained and saved in the
`models` object. These will be be compared to predictions from the *ensemble* later.

```r
models$pred$validation <- sapply(seq(1, nmodels), 
                                 function(x) predict(models$fit[[x]], validation))
models$pred$validation <- as.data.frame(models$pred$validation)
colnames(models$pred$validation) <- models$name
```

#### Confusion Matrices
The confusion matrices corresponding to each model are obtained for the validation
data set. Again, these are saved in the `models` object for comparison with the 
corresponding confusion matrix from the *ensemble* model.

```r
models$confmat  <- 
  sapply(seq(1, nmodels),
         function(x) 
           list(confusionMatrix(models$pred$validation[, x], validation$classe)))

models$confmat$summary <- sapply(seq(1, nmodels), 
                                 function(x) models$confmat[[x]]$overall)
models$confmat$summary <- as.data.frame(models$confmat$summary)
colnames(models$confmat$summary) <- models$name
```

### Ensemble: Combined Predictors
The *ensemble* model "stacks" the predictors obtained from the four machine learning
models used before. The fit, predictions on the validation set and the confusion
matrix are all stored in the `models$ensemble` object.

```r
ensemble.df <- cbind(models$pred$validation, classe = validation$classe)
models$ensemble$fit <- FitModel(ensemble.df, model.method = "rf")[[1]]

models$ensemble$pred$validation <- 
  predict(models$ensemble$fit, models$pred$validation)
models$ensemble$confmat <- 
  confusionMatrix(models$ensemble$pred$validation, validation$classe)
```

## Comparison
****
All models can now be compared. Since this is a classification problem, 
accuracy, sensitivity and specificity are used as metrics to assess the prediction
quality of each model.  

### Accuracy
As seen below, the *kNearestNeighbors* has the highest accuracy of individual 
models but the *ensemble* model has the highest accuracy. The lower and upper
bounds on accuracy are also summarized in the table and are tightest for the ensemble model.

```r
library(xtable)
confmat.summary <- cbind(models$confmat$summary, 
                         Ensemble = models$ensemble$confmat$overall)
print(xtable(confmat.summary), type = "html")
```

<!-- html table generated in R 3.1.0 by xtable 1.7-3 package -->
<!-- Sun Jul 20 21:46:37 2014 -->
<TABLE border=1>
<TR> <TH>  </TH> <TH> RandomForest </TH> <TH> BaggedCART </TH> <TH> SVMRadial </TH> <TH> kNearestNeighbors </TH> <TH> Ensemble </TH>  </TR>
  <TR> <TD align="right"> Accuracy </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 0.79 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> Kappa </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 0.94 </TD> <TD align="right"> 0.73 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> AccuracyLower </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 0.78 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> AccuracyUpper </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 0.80 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> AccuracyNull </TD> <TD align="right"> 0.28 </TD> <TD align="right"> 0.28 </TD> <TD align="right"> 0.28 </TD> <TD align="right"> 0.28 </TD> <TD align="right"> 0.28 </TD> </TR>
  <TR> <TD align="right"> AccuracyPValue </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 0.00 </TD> </TR>
  <TR> <TD align="right"> McnemarPValue </TD> <TD align="right">  </TD> <TD align="right"> 0.00 </TD> <TD align="right"> 0.00 </TD> <TD align="right">  </TD> <TD align="right">  </TD> </TR>
   </TABLE>

### Sensitivity
For each class, *kNearestNeighbors* and *ensemble* sensitivities are comparable though
the *ensemble* model seems to be *very* slightly better than *kNearestNeighbors* and the
best overall.

```r
library(xtable)
sensitivity.summary <- 
  cbind(sapply(seq(1, nmodels), function(x) models$confmat[[x]]$byClass[, 1]), 
      models$ensemble$confmat$byClass[, 1])
colnames(sensitivity.summary) <- c(models$name, "Ensemble")
print(xtable(sensitivity.summary), type = "html")
```

<!-- html table generated in R 3.1.0 by xtable 1.7-3 package -->
<!-- Sun Jul 20 21:46:37 2014 -->
<TABLE border=1>
<TR> <TH>  </TH> <TH> RandomForest </TH> <TH> BaggedCART </TH> <TH> SVMRadial </TH> <TH> kNearestNeighbors </TH> <TH> Ensemble </TH>  </TR>
  <TR> <TD align="right"> Class: A </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> Class: B </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 0.93 </TD> <TD align="right"> 0.89 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> Class: C </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 0.72 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.98 </TD> </TR>
  <TR> <TD align="right"> Class: D </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 0.94 </TD> <TD align="right"> 0.37 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.99 </TD> </TR>
  <TR> <TD align="right"> Class: E </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.96 </TD> <TD align="right"> 0.81 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.99 </TD> </TR>
   </TABLE>

### Specificity
Again, the *ensemble* model seems to be have the highest specificity for each class
when compared to all other models.

```r
library(xtable)
specificity.summary <- 
  cbind(sapply(seq(1, nmodels), function(x) models$confmat[[x]]$byClass[, 2]), 
      models$ensemble$confmat$byClass[, 2])
colnames(specificity.summary) <- c(models$name, "Ensemble")
print(xtable(specificity.summary), type = "html")
```

<!-- html table generated in R 3.1.0 by xtable 1.7-3 package -->
<!-- Sun Jul 20 21:46:37 2014 -->
<TABLE border=1>
<TR> <TH>  </TH> <TH> RandomForest </TH> <TH> BaggedCART </TH> <TH> SVMRadial </TH> <TH> kNearestNeighbors </TH> <TH> Ensemble </TH>  </TR>
  <TR> <TD align="right"> Class: A </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.94 </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 1.00 </TD> </TR>
  <TR> <TD align="right"> Class: B </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.93 </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 1.00 </TD> </TR>
  <TR> <TD align="right"> Class: C </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.98 </TD> <TD align="right"> 0.94 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 1.00 </TD> </TR>
  <TR> <TD align="right"> Class: D </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.97 </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 1.00 </TD> </TR>
  <TR> <TD align="right"> Class: E </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 0.99 </TD> <TD align="right"> 0.95 </TD> <TD align="right"> 1.00 </TD> <TD align="right"> 1.00 </TD> </TR>
   </TABLE>

## Predictions on Testing Data
****
All five models are used to predict the `classe` variable for the 20 observations
in the testing data set.

```r
models$pred$testing    <- sapply(seq(1, nmodels), 
                                 function(x) predict(models$fit[[x]], testing))
models$pred$testing <- as.data.frame(models$pred$testing)
colnames(models$pred$testing) <- models$name

models$ensemble$pred$testing <- predict(models$ensemble$fit, models$pred$testing)
```

The table below shows that all models predict the exact same outcome for 
all the observations, except for the 3rd observation by *RandomForest* and the
3rd and 11th observations by *SVMRadial*. One can therefore have high confidence 
in the predictions.

```r
prediction.testing <- cbind(models$pred$testing, 
                            Ensemble = models$ensemble$pred$testing)
print(xtable(t(prediction.testing)), type = "html")
```

<!-- html table generated in R 3.1.0 by xtable 1.7-3 package -->
<!-- Sun Jul 20 21:47:46 2014 -->
<TABLE border=1>
<TR> <TH>  </TH> <TH> 1 </TH> <TH> 2 </TH> <TH> 3 </TH> <TH> 4 </TH> <TH> 5 </TH> <TH> 6 </TH> <TH> 7 </TH> <TH> 8 </TH> <TH> 9 </TH> <TH> 10 </TH> <TH> 11 </TH> <TH> 12 </TH> <TH> 13 </TH> <TH> 14 </TH> <TH> 15 </TH> <TH> 16 </TH> <TH> 17 </TH> <TH> 18 </TH> <TH> 19 </TH> <TH> 20 </TH>  </TR>
  <TR> <TD align="right"> RandomForest </TD> <TD> B </TD> <TD> A </TD> <TD> C </TD> <TD> A </TD> <TD> A </TD> <TD> E </TD> <TD> D </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> B </TD> <TD> C </TD> <TD> B </TD> <TD> A </TD> <TD> E </TD> <TD> E </TD> <TD> A </TD> <TD> B </TD> <TD> B </TD> <TD> B </TD> </TR>
  <TR> <TD align="right"> BaggedCART </TD> <TD> B </TD> <TD> A </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> E </TD> <TD> D </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> B </TD> <TD> C </TD> <TD> B </TD> <TD> A </TD> <TD> E </TD> <TD> E </TD> <TD> A </TD> <TD> B </TD> <TD> B </TD> <TD> B </TD> </TR>
  <TR> <TD align="right"> SVMRadial </TD> <TD> B </TD> <TD> A </TD> <TD> C </TD> <TD> A </TD> <TD> A </TD> <TD> E </TD> <TD> D </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> A </TD> <TD> C </TD> <TD> B </TD> <TD> A </TD> <TD> E </TD> <TD> E </TD> <TD> A </TD> <TD> B </TD> <TD> B </TD> <TD> B </TD> </TR>
  <TR> <TD align="right"> kNearestNeighbors </TD> <TD> B </TD> <TD> A </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> E </TD> <TD> D </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> B </TD> <TD> C </TD> <TD> B </TD> <TD> A </TD> <TD> E </TD> <TD> E </TD> <TD> A </TD> <TD> B </TD> <TD> B </TD> <TD> B </TD> </TR>
  <TR> <TD align="right"> Ensemble </TD> <TD> B </TD> <TD> A </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> E </TD> <TD> D </TD> <TD> B </TD> <TD> A </TD> <TD> A </TD> <TD> B </TD> <TD> C </TD> <TD> B </TD> <TD> A </TD> <TD> E </TD> <TD> E </TD> <TD> A </TD> <TD> B </TD> <TD> B </TD> <TD> B </TD> </TR>
   </TABLE>


## Concluding Remarks
****
Four machine learning algorithms, RandomForest, BaggedCART, SVMRadial and 
kNearestNeighbors were used to model the training set. An ensemble model was created
using the predictions from the four algorithms. It was found that the *kNearestNeighbors*
has the highest accuracy of the individual algorithms but the *ensemble* model performed
slightly better overall. The ensemble model is used below as answers to predictions on the testing set.


```r
answers <- as.vector(prediction.testing$Ensemble)

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("submission/problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```


## Document Information
****
This document was generated using knitr and pandoc

`knit('project.Rmd')`  

`system("pandoc project.md -o project.html -c custom.css --toc --self-contained --highlight-style=tango")`


```r
sessionInfo()
```

```
R version 3.1.0 (2014-04-10)
Platform: x86_64-w64-mingw32/x64 (64-bit)

locale:
[1] LC_COLLATE=English_United States.1252 
[2] LC_CTYPE=English_United States.1252   
[3] LC_MONETARY=English_United States.1252
[4] LC_NUMERIC=C                          
[5] LC_TIME=English_United States.1252    

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] xtable_1.7-3       plyr_1.8.1         e1071_1.6-3       
 [4] kknn_1.2-5         kernlab_0.9-19     ipred_0.9-3       
 [7] randomForest_4.6-7 caret_6.0-30       ggplot2_0.9.3.1   
[10] lattice_0.20-29    knitr_1.6.8       

loaded via a namespace (and not attached):
 [1] BradleyTerry2_1.0-5 brglm_0.5-9         car_2.0-20         
 [4] class_7.3-10        codetools_0.2-8     colorspace_1.2-4   
 [7] compiler_3.1.0      digest_0.6.4        evaluate_0.5.5     
[10] foreach_1.4.2       formatR_0.10        grid_3.1.0         
[13] gtable_0.1.2        gtools_3.4.1        igraph_0.7.0       
[16] iterators_1.0.7     lava_1.2.6          lme4_1.1-6         
[19] MASS_7.3-31         Matrix_1.1-3        minqa_1.2.3        
[22] munsell_0.4.2       nlme_3.1-117        nnet_7.3-8         
[25] prodlim_1.4.3       proto_0.3-10        Rcpp_0.11.1        
[28] RcppEigen_0.3.2.1.2 reshape2_1.4        rpart_4.1-8        
[31] scales_0.2.4        splines_3.1.0       stringr_0.6.2      
[34] survival_2.37-7     tools_3.1.0        
```

