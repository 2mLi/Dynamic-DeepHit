# This script matches WGS analysis data with PreCar

Train <- read.csv("data/precar_Train_plus.csv", header = T)
Test <- read.csv("data/precar_Test_plus.csv", header = T)
nrow(Train) + nrow(Test)
# 12701
# read WGS data
wgs <- read.delim("data/auxillary_data/WGS_HIFI.txt")
wgs$Sample <- substr(wgs$Sample, 3, 11)

# this converts CNV_Score
wgs$CNV_score <- ifelse(wgs$CNV_score == 0, 0, 1)

# match up
library(dplyr)
Train_new <- left_join(Train, wgs, by = c("Sample"))
Test_new <- left_join(Test, wgs, by = c("Sample"))

Test_new$Inr <- as.numeric(Test$Inr)

write.csv(Train_new, "data/precar_Train_WGS_plus.csv", row.names= F, fileEncoding = "UTF-8")

write.csv(Test_new, "data/precar_Test_WGS_plus.csv", row.names= F, fileEncoding = "UTF-8")
