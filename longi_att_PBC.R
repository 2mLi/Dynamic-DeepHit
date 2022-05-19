


# subset data to remove long-term patients

Train <- read.csv("data/PBC2_Train_rnd.csv", header = T)
Test <- read.csv("data/PBC2_Test_rnd.csv", header = T)

tr_lt <- read.table("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PBC/2022-05-11_16-10-49-353454_PBC_model/eval/tr_longterm_id.txt", quote="\"", comment.char="")
te_lt <- read.table("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PBC/2022-05-11_16-10-49-353454_PBC_model/eval/te_longterm_id.txt", quote="\"", comment.char="")

Train_st <- Train[!(Train$ID %in% tr_lt[, 1]), ]
Test_st <- Test[!(Test$ID %in% te_lt[, 1]), ]

# export
write.csv(Train_st, "data/PBC2_Train_rnd_lt.csv", row.names = F)
write.csv(Test_st, "data/PBC2_Test_rnd_lt.csv", row.names = F)