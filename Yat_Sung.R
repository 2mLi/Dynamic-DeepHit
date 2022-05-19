# Yat-Sung Uni's landmark approach

library(survival)

# step 1: univariate cox, using a bonferonni's corrected p threshold

# step 2: sample size check
# - as a simple repetition, we could just use 20 * number of predictors as a benchmark

# step 3: multivariate cox, then backward elimination until one variable remains

# this would be a multi-layer for loop? 

data_slicer <- function(dat, 
                        time_interval, 
                        time, 
                        id, 
                        vars, 
                        min_time = NULL, 
                        max_time = NULL) {
  # this slicer can only slice data into regular intervals, like the one in paper
  # the result was stored in a 3-d matrix like DDH
  # but I expect the user could simply extract by one dimension and convert it back
  # to a R-base dataframe
  
  # create time slices
  # by default, if no min and max specified, will use min and max of dat$time
  
  # for some reasons, vars must be greater than length 2
  # for this version of the code, let us ignore this problem...
  require(survival)
  
  if (is.null(min_time)){
    min_time <- min(dat[, time])
  }
  
  if (is.null(max_time)){
    max_time <- max(dat[, time])
  }
  
  # time slice is created by dividing [min, max] by time_slice
  # here each number specifies the 'starting point' of time slice
  time_slice <- seq(from = min_time, to = max_time, by = time_interval)
  
  
  
  # dim_spec: 1st dim is patient ids, 2nd dim time slice, 3rd dim data at each slice
  
  # remember principle: 
  # - if multiple data at the same slice: take the latest one
  # - if no data at any slice: should be all NA
  # - as a bonus, I'll add one vector at 3rd dim specifying the status at this slice
  # - - 0: no data; 1: one data; 2: more than one data. 
  
  # user must specify what variables to be included, in vars
  pat_list <- unique(dat[, id])
  
  dim_spec = c(length(pat_list), length(time_slice), length(vars) + 1)
  dimname_1 <- c()
  for (i in pat_list){
    dimname_1 <- c(dimname_1, toString(i))
  }
  dimname_2 <- c()
  for (i in time_slice){
    dimname_2 <- c(dimname_2, toString(i))
  }
  dimname_3 <- c("mi", vars)
  
  
  output <- array(NA, dim = dim_spec, dimnames = list(dimname_1, dimname_2, dimname_3))
  
  # fill in output 
  for (i in 1:dim_spec[1]){
    for (j in 1:dim_spec[2]){
      
      min_range <- time_slice[j]
      max_range <- time_slice[j] + time_interval
      
      # subset data by patient name and min, max range
      idx_range <-(dat[, time] >= min_range) & (dat[, time] < max_range)
      pat_range <- (dat[, id] == pat_list[i])
      idx <- idx_range & pat_range
      dat_sub <- dat[idx, vars]
      t <- dat[idx, time]
      
      # now we want to see how many rows dat_sub have, and take corresponding actions
      if (nrow(dat_sub) == 0){
        # no data
        output[dimname_1[i], dimname_2[j], 1] <- 0
        # remaining elements marked as NA
      }else if (nrow(dat_sub) == 1){
        output[dimname_1[i], dimname_2[j], 1] <- 1
        output[dimname_1[i], dimname_2[j], -1] <- dat_sub
      }else if (nrow(dat_sub) >= 2){
        # multiple records found
        # use the newest one specified by time
        target_dat <- dat_sub[which.max(t), ]
        output[dimname_1[i], dimname_2[j], 1] <- 2
        output[dimname_1[i], dimname_2[j], -1] <- target_dat
        dim(output) <- dim_spec
        dimnames(output) <- list(dimname_1, dimname_2, dimname_3)
      }
      
    }
  }
  
  return(output)
}

f_step_1 <- function(dat, 
                     predictor, 
                     times, 
                     status, 
                     p_thr = 0.05, 
                     b_corr = TRUE) { 
  
  # this function fits univariate cox for each specified predictors in dat
  # for now, we assume dat comes from ONE SINGLE time slice ONLY
  # and has been carefully prepared
  # so this function, on its own, does not check data quality issues
  p_L <- rep(NA, length(predictor))
  for (i in 1:length(predictor)){
    # y <- dat[, response]
    x <- dat[, predictor[i]]
    t <- dat[, times]
    s <- dat[, status]
    
    mod <- coxph(Surv(t, s) ~ x)
    this_p <- summary(mod)$coefficients[, 5]
    p_L[i] <- this_p
  }
  
  if (b_corr == TRUE){
    p_thr <- p_thr / length(predictor) # Bon's correction
  }
  
  selected_predictor <- predictor[p_L < p_thr]
  
  return(selected_predictor)
}



f_step_3 <- function(dat, 
                     times, 
                     status, 
                     p_thr = 0.05){
  # fit a multivariate regression model
  # assumes predictors were dat's all inputs, except status and times
  predictors <- names(dat)
  predictors <- predictors[predictors != times & predictors != status]
  
  repeat{
    
    # cat("Testing: ", paste(predictors, ","), "\n", sep = "")
    mod <- coxph(Surv(Times, Status) ~ ., data = dat)
    # find the one with the largest p value
    ps <- summary(mod)$coefficients[, 5]
    # cat(ps)
    eliminated_var <- names(ps[which.max(ps)])
    # cat(eliminated_var)
    # cat("\n")
    predictors <- predictors[predictors != eliminated_var]
    # cat(predictors)
    dat <- dat[, c(predictors, times, status)]
    # break when only one predictor is left
    if (length(predictors) <= 1){
      break
    }
  }
  return(predictors)
}

