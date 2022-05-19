# let us see how xgboost does
library(caret)
library(xgboost)

Train <- read.csv("data/precar_Train_WGS_plus.csv", header = T)
Test <- read.csv("data/precar_Test_WGS_plus.csv", header = T)
source("my_functions.R", encoding = "UTF-8")

# make Train into Train_w
Train_w <- toSurv(Train, idvar = "Patient", timevar = "Time", newtimevar = "Times")
Test_w <- toSurv(Test, idvar = "Patient", timevar = "Time", newtimevar = "Times")

Train_w_label <- Train_w$Status

grid = expand.grid(
  nrounds = seq(50, 100, 5), 
  eta = c(0.01, 0.1, 0.3, 0.5, 0.6), 
  gamma = c(0.1, 0.25, 0.5), 
  max_depth = c(0.25, 0.5, 0.75, 1)
)
set.seed(12345)

cntrl = trainControl(method = "cv", 
                     number = 10, 
                     verboseIter = FALSE, 
                     returnData = FALSE, 
                     returnResamp = "final")
train.xgb = train(
  x = Train_w[, c(3, 4, 6, 7:10, 12:19, 21:23, 29:35)], 
  y = ,Train_w_label, 
  trControl = cntrl, 
  tuneGrid = grid, 
  
)
params = list(
  objective = "survival:cox", 
  eval_metric = "logloss", 
  booster = "gbtree", 
  eta = "0.1", 
  max_depth = 2, 
  subsample = 0.5, 
  colsample_bytree = 1, 
  gamma = 0.1
)

Train <- read.csv("data/precar_Train_WGS_plus.csv", header = T)
Test <- read.csv("data/precar_Test_WGS_plus.csv", header = T)
source("my_functions.R", encoding = "UTF-8")

# make Train into Train_w
Train_w <- toSurv(Train, idvar = "Patient", timevar = "Time", newtimevar = "Times")
Test_w <- toSurv(Test, idvar = "Patient", timevar = "Time", newtimevar = "Times")

# make that into the matrix that xgb wants
Train_w_mat <- as.matrix(Train_w[, c(3, 4, 6, 7:10, 12:19, 21:23, 29:35)])
Train_w_label <- ifelse(Train_w$Status == 1, Train_w$Times, -Train_w$Times)

Test_w_mat <- as.matrix(Test_w[, c(3, 4, 6, 7:10, 12:19, 21:23, 29:35)])
Test_w_label <- ifelse(Test_w$Status == 1, Test_w$Times, -Test_w$Times)

Train_xgb_input <- xgb.DMatrix(data = Train_w_mat, 
                               label = Train_w_label, 
                               )
Test_xgb_input <- xgb.DMatrix(data = Test_w_mat, 
                               label = Test_w_label, 
)

set.seed(142857)
mod <- xgb.train(params = params, 
                 data = Train_xgb_input, 
                 nrounds = 100, 
                 verbose = 2, 
                 print_every_n = 5)
# prediction
pred_int <- predict(mod, Train_xgb_input)
pred_ext <- predict(mod, Test_xgb_input)

# importance matrix
impMatrix <- xgb.importance(feature_names = dimnames(Train_xgb_input)[[2]], model = mod)
xgb.plot.importance(impMatrix)

# ROC and confusion matrix
library(InformationValue)
cutoff <- optimalCutoff(Train_w_label, pred_int)
cat("Optimal cutoff is: ", cutoff, sep = "")

# c and B
library(Hmisc)
library(riskRegression)

xt <- seq(from = -10, to = 10, by = 0.01)
yt <- dt(xt, df = 4)
dat <- data.frame(xt = xt, yt = yt)
t <- 3.0339
critT <- 2.7764
dat_crit1 <- dat[xt >= critT, ]
dat_crit2 <- dat[xt <= -critT, ]
library(ggplot2)
ggplot() + 
  theme_bw() + 
  geom_line(data = dat, aes(x = xt, y = yt)) + 
  labs(x = "T", y = "Density") + 
  geom_ribbon(data = dat_crit1, aes(x = xt, ymin = 0, ymax = yt), fill = "pink") + 
  geom_ribbon(data = dat_crit2, aes(x = xt, ymin = 0, ymax = yt), fill = "pink") + 
  geom_vline(xintercept = t, linetype = "dashed")

# kappa test

