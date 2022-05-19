# make random disturbances on PBC2 data

Train <- read.csv("data/PBC2_Train_rnd.csv")

# make some disturbances on year-6 onwards

# using serBilir? 
l <- length(Train$serBilir[Train$Times >= 6])
Train$serBilir[Train$Times >= 6] <- Train$serBilir[Train$Times >= 6] + rnorm(n = l, mean = 0, sd = 0.4)

write.csv(Train, "data/PBC2_Train_rnd_rand.csv", row.names = F)