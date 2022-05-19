model_name <- "PreCar/2022-03-28_02-10-21-191288_my_aMAP_model_with_CNVs"

path <- paste(model_name, "/eval/_longi_att__.txt", sep = "")
longi_att <- read.csv(path, header=FALSE)

co <- ncol(longi_att)

# colnames(longi_att) <- paste("FU", seq(0, 14, 1), sep = " ")

longi_att$BFU <- NA
for (i in 1:nrow(longi_att)){
  longi_att$BFU[i] <- which.max(longi_att[i, 1:co])
}

print(table(longi_att$BFU))


# subset data to remove long-term patients

Train <- read.csv("data/PreCar_Train_WGS_plus.csv", header = T)
Test <- read.csv("data/PreCar_Test_WGS_plus.csv", header = T)

tr_lt <- read.table("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/2022-03-28_02-10-21-191288_my_aMAP_model_with_CNVs/eval/tr_longterm_id.txt", quote="\"", comment.char="")
te_lt <- read.table("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/2022-03-28_02-10-21-191288_my_aMAP_model_with_CNVs/eval/te_longterm_id.txt", quote="\"", comment.char="")

Train_st <- Train[!(Train$ID %in% tr_lt[, 1]), ]
Test_st <- Test[!(Test$ID %in% te_lt[, 1]), ]

# export
write.csv(Train_st, "data/PreCar_Train_WGS_plus_shortTerm.csv", row.names = F)
write.csv(Test_st, "data/PreCar_Test_WGS_plus_shortTerm.csv", row.names = F)

# what is the difference in longitudinal trends between st and lt patients? 

Train_lt <- Train[(Train$ID %in% tr_lt[, 1]), ]
Test_lt <- Test[(Test$ID %in% te_lt[, 1]), ]

precar_st <- rbind(Train_st, Test_st)
precar_lt <- rbind(Train_lt, Test_lt)

library(mixAK)
ip_st <- getProfiles(t = 'Time', y = c('Afp', 'Age', 'Alb', 'Plt', 'Tb', 'Inr'), 
                  id = 'Patient', 
                  data = precar_st)
ip_lt <- getProfiles(t = 'Time', y = c('Afp', 'Age', 'Alb', 'Plt', 'Tb', 'Inr'), 
                     id = 'Patient', 
                     data = precar_lt)
xLim <- c(0, 36)

cont_list <- c('Afp', 'Age', 'Alb', 'Plt', 'Tb', 'Inr')

for (varname in cont_list){
  
  plotProfiles(ip_st, data = precar_st, xlim = xLim, var = varname, tvar = 'Time', 
               main = paste('short term,', varname))
  
  plotProfiles(ip_lt, data = precar_lt, xlim = xLim, var = varname, tvar = 'Time', 
               main = paste('long term,', varname))
  
}
  
