# library(survival)

source("Yat_Sung.R")
library(survminer)
library(survival)

Train <- read.csv("data/precar_Train_WGS_plus.csv", header = T)

Test <- read.csv("data/precar_Test_WGS_plus.csv", header = T)

# Train <- read.csv("data/PBC2_Train_rnd.csv", header = T)

# Test <- read.csv("data/PBC2_Test_rnd.csv", header = T)

interv <- 6

PBC2_vars <- c("serBilir", "serChol", "albumin", 
               "alkaline", 
               "SGOT", 
               "platelets", 
               "prothrombin", 
               "Status", 
               "Times", 
               "ID")

PreCar_vars <- c("Age", "Alb", "Plt", "Tb", "Afp", "HIFI", 
                 "Alp", "Alt", "Inr", 
                 "Status", 
                 "Times", 
                 "ID")

Train_s <- data_slicer(Train, interv, 
                       "Time", 
                       "ID", 
                       PreCar_vars
                       )

Test_s <- data_slicer(Test, interv, 
                       "Time", 
                       "ID", 
                      PreCar_vars
)

# for each time slice ...

time_slice <- seq(from = min(Train$Time), to = max(Train$Time), 
                  by = interv)

selected_predictors_step_1 <- vector(mode = "list", length = 1)
selected_predictors_step_3 <- vector(mode = "list", length = 1)
thr_step_3 <- vector(mode = "list", length = 1)
c_idx_3 <- vector(mode = "list", length = 1)

Grp <- data.frame(ID = unique(Train$ID), Grp1 = NA, Grp2 = NA, Grp3 = NA)
for (i in 1:1){
  Train_sub <- as.data.frame(Train_s[, i, ])
  for (k in 1:ncol(Train_sub)){
    Train_sub[, k] <- as.numeric(Train_sub[, k])
  }
  
  # adjust for landmark time
  Train_sub[, "Times"] <- Train_sub[, "Times"] - time_slice[i]
  
  # apply f_step_1
  predictors <- names(Train_sub)
  predictors <- predictors[predictors != "mi"]
  for (pred in predictors){
    Train_sub[, pred] <- as.numeric(Train_sub[, pred])
  }
  # remove Times, Status and ID
  predictors <- predictors[predictors != "Status" & predictors != "ID" & predictors != "Times"]

  
  cat("Step 1: \n")
  selected_predictors <- f_step_1(Train_sub, 
                                 predictor = predictors, 
                                 times = "Times", 
                                 status = "Status", 
                                 p_thr = 0.5
                                 )
  if (length(selected_predictors) > 0){
    selected_predictors_step_1[[i]] <- selected_predictors
    
    # proceed to step 2
    cat("Step 2: \n")
    # next, step 2
    # its just a condition: does sample size meet 20 * selected_predictors? 
    
    
    if (nrow(Train_sub[, selected_predictors]) >= 20* length(selected_predictors)){
      
      # proceed to step 3
      ID <- Train_sub[, "ID"]
      Train_sub <- Train_sub[, c(selected_predictors, "Status", "Times")]
      
      # use a custom function
      cat("Step 3: \n")
      selected_var <- f_step_3(Train_sub, "Times", "Status")
      
      # this is our final decision for this time slice
      selected_predictors_step_3[[i]] <- selected_var
      
      # assuming all above were proceeding as normal, we should find a best threshold here
      
      # first, fit a model with selected_var
      Train_sub <- Train_sub[, c(selected_var, "Status", "Times")]
      mod <- coxph(Surv(Times, Status) ~ ., data = Train_sub)
      
      proba <- predict(mod, Train_sub, type="risk")
      
      Train_sub$risk <- proba
      
      #选择最优阈值, at this time slice
      res_cut <- surv_cutpoint(Train_sub, time = "Times", event = "Status",variables = c("risk"))
      cutoff <- res_cut$risk$estimate
      
      thr_step_3[[i]] <- cutoff
      
      # find c-index, which is hidden in mod
      c_idx_3[[i]] <- summary(mod)$concordance
      
      # finally, the ultimate challenge... 
      # assign groupings based on cutoff, then loop back
      group_name <- paste("Group", i, sep = "")
      Grp_1<- ifelse(Train_sub$risk >= cutoff, 1, 0)
      
      
      
      
      
      
    }
    
  }
  
  
  
}

# the idea is as follows: we use Grp_1 to manually do the next separation

# for the next: 

