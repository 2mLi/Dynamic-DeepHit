# segregate PBC2 dataset by years

Train <- read.csv("data/PBC2_Train_rnd.csv")
Test <- read.csv("data/PBC2_Test_rnd.csv")

# first, process data's binary variables
Train$Ascites = ifelse(Train$ascites == "Yes", 1, 0)
Train$Hepatomegaly = ifelse(Train$hepatomegaly == "Yes", 1, 0)
Train$Ascites = ifelse(Train$ascites == "Yes", 1, 0)
# max year is like 14 years

Train_yr4 <- Train[Train$years <= 4, ]
Test_yr4 <- Train[Train$years <= 4, ]

Train_yr8 <- Train[Train$years <= 8, ]
Test_yr8 <- Train[Train$years <= 8, ]

write.csv(Train_yr4, "data/PBC2_Train_yr4.csv", row.names = F)
write.csv(Test_yr4, "data/PBC2_Test_yr4.csv", row.names = F)

write.csv(Train_yr8, "data/PBC2_Train_yr8.csv", row.names = F)
write.csv(Test_yr8, "data/PBC2_Test_yr8.csv", row.names = F)


# and we investigate the distribution of Status over different time intervals
# interval lengths: c(1, 2, 5, 10)

pred_time <- c(1, 3, 5, 10)
eval_time <- seq(from = 0, to = 14, by = 1)

# create a table that stores results
res <- matrix(NA, nrow = length(pred_time), ncol = length(eval_time))
rownames(res) <- pred_time
colnames(res) <- eval_time

risk <- matrix(NA, nrow = length(pred_time), ncol = length(eval_time))
rownames(risk) <- pred_time
colnames(risk) <- eval_time

# fill in matrix
# for train, we need: Status and Times
Status <- Train$Status
Times <- Train$Times
for (i in 1:length(pred_time)){
  for (j in 1:length(eval_time)){
    
    HCC <- sum(Status == 1 & Times >= eval_time[j] & Times < eval_time[j] + pred_time[i])
    # total surviving patient at this time
    total <- sum(Times >= eval_time[j])
    
    res[i, j] <- paste(HCC, total, sep = "/")
    risk[i, j] <- round(HCC/total, digits =4)
    
  }
}

c_index <- read.csv("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PBC/2022-05-11_04-34-49-826605_PBC_model/eval/Train_c_index.csv", 
                    row.names=1)
rownames(c_index) <- pred_time
colnames(c_index) <- eval_time


res
risk
c_index