# read pbc2 data
library(JMbayes2)
data(pbc2)

# change names
pbc2$ID <- pbc2$id

pbc2$Times <- pbc2$years

pbc2$Time <- pbc2$year

pbc2$Status <- ifelse(pbc2$status == "alive", 0, 1)

pbc2$Sex <- ifelse(pbc2$sex == "female", 0, 1)

pbc2$Drug <- ifelse(pbc2$drug == "placebo", 0, 1)

# introduce ALBI
pbc2$AlBi <- 0.66 * log(pbc2$serBilir + 0.0001, 10) - 0.085 * pbc2$albumin

# here, two types of train/test split: 
# 1) random split
set.seed(6324)
id <- unique(pbc2$ID)
id_tr <- sample(id, size = round(0.5*length(id)))
PBC2_Train_rnd <- pbc2[pbc2$ID %in% id_tr, ]
PBC2_Test_rnd <- pbc2[!(pbc2$ID %in% id_tr), ]

# to export
write.csv(PBC2_Train_rnd, "data/PBC2_Train_rnd.csv", row.names = F, fileEncoding = "UTF-8")
write.csv(PBC2_Test_rnd, "data/PBC2_Test_rnd.csv", row.names = F, fileEncoding = "UTF-8")