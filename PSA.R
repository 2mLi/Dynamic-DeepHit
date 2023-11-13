# import data

library(dplyr)

file_trajectory <- paste0('/home/zhengmaoli/Documents/Workspace/Slingshot_Tryout', '/data/')

dat <- read.csv(paste0(file_trajectory,  'all_psa_data.csv'), header = T)

dat$patient_no <- as.character(dat$patient_no)

# exploratory

names(dat) # there are many data, only a few is relevant

# specifically: time_since_first_psa is longitudinal time; psa_measurement is 
# index of measurement; psa_value is absolute value of psa; 
# patient_no is patient ID; 
# risk is pre-T risk group; 
# psa_recurrence; 
# followup_years is max follow up years

# how many follow-ups does each patient have? 

hist(table(dat$patient_no), breaks = 70)

min(table(dat$patient_no)) # minimum: 2 follow-ups
max(table(dat$patient_no)) # maximum: 67 follow-ups??

# PSA trajectory of random patients

library(ggplot2)

seed <- 1024
rnd_patient_no <- sample(unique(dat$patient_no), 20)

dat_sub <- dat %>% filter(patient_no %in% rnd_patient_no)

gplot <- ggplot(dat_sub) + geom_line(aes(time_since_first_psa, log_psa_value, group=patient_no, colour=patient_no)) 



gplot

# what is wrong with those missing values? 

# anyway the relationship is very non-linear. 

# can we try eg mixAK on stable PSA patients? 




