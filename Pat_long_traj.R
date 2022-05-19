# import data
Test <- read.csv("data/precar_Test.csv", header = T)

# potential candidate: 4044
# read json
library(rjson)

# read json file
tag = "2022-03-23_03-01-20-269822_my_aMAP_model"
file_name1 = "L0H6_log.json"
dir1 = paste("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/"
 , tag
 , "/eval/pat_long_traj/"
 , file_name1, sep = "")
file_name2 = "L5H6_log.json"
dir2 = paste("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/"
             , tag
             , "/eval/pat_long_traj/"
             , file_name2, sep = "")
# file_name3 = "L11H6_log.json"
# dir3 = paste("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/"
#              , tag
#              , "/eval/pat_long_traj/"
#              , file_name3, sep = "")
# file_name4 = "L26H6_log.json"
# dir4 = paste("F:/Anaconda3/envs/DDH/Dynamic-DeepHit/PreCar/"
#              , tag
#              , "/eval/pat_long_traj/"
#              , file_name4, sep = "")
result1 <- fromJSON(file = dir1)
result2 <- fromJSON(file = dir2)
result3 <- fromJSON(file = dir3)
result4 <- fromJSON(file = dir4)

# patient longi profile
pat <- result1$patient_idx
datDat <- Test[Test$ID == 4044, ]
t1 <- result1$t
risk1 <- result1$risk
t2 <- result2$t
risk2 <- result2$risk
# t3 <- result3$t
# risk3 <- result3$risk
# t4 <- result4$t
# risk4 <- result4$risk

# plot
library(ggplot2)
fac = 100
ymax <- max(log(datDat$Afp + 0.0001), fac * c(risk1, risk2))
idx1 <- t1 < min(t2)
ggplot() + 
  theme_classic() + 
  geom_point(aes(x = datDat$Time, y = log(datDat$Afp + 0.0001, 10)), colour = "black") + 
  geom_line(aes(x = datDat$Time, y = log(datDat$Afp + 0.0001, 10)), colour = "red") + 
  geom_line(aes(x = t1[idx1], y = fac * risk1[idx1]), colour = "blue") +
  geom_line(aes(x = t2, y = fac * risk2), colour = "blue") +
  # geom_line(aes(x = t3, y = fac * risk3), colour = "blue") +
  # geom_line(aes(x = t4, y = fac * risk4), colour = "blue") +
  # geom_area(aes(x = t1, y = fac * risk1), fill= "lightblue") +
  # geom_area(aes(x = t2, y = fac * risk2), fill= "lightblue") +
  # geom_area(aes(x = t3, y = fac * risk3), fill= "lightblue") +
  # geom_area(aes(x = t4, y = fac * risk4), fill= "lightblue") +
  # geom_vline(xintercept = 7, linetype = 2) +
  # geom_vline(xintercept = 13, linetype = 2) +
  # geom_vline(xintercept = 19, linetype = 2) +
  # geom_vline(xintercept = 25, linetype = 2) +
  scale_y_continuous(
    name = "log(Afp)", 
    limits = c(0, ymax + 0.1), 
    sec.axis = sec_axis( trans=~.*1/fac, name = "Risk")
  ) + 
  scale_x_continuous(
    name = "Time (Month)"
  )