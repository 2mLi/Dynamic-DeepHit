library(ggplot2)
library(dplyr)

# setwd("/Users/yanghao/BerryOncology/甲基化/meCap/Fragment_Methylated_Reads")
dat <- read.table("Hao_fragment.txt",header=T)
dat <- dat %>% mutate(type = as.factor(type))
cdat <- dat %>% group_by(type) %>% filter(frequency >= max(frequency)) %>% select(type, fragment) %>% rename(max_frag = fragment)

ggplot(dat,aes(x = fragment,y = frequency, colour = type))+
  geom_line(size=2,alpha=1) +
  geom_vline(data=cdat, aes(xintercept=max_frag), linetype="dashed",size=1) +
  geom_text(data = cdat, aes(x = max_frag, label = max_frag), 
            y = 0.001, angle = 90, vjust = -0.4, colour = "black") +
  facet_wrap(~type,scales= "free",ncol=1) +
  #geom_vline(xintercept = c(166),size=1)+
  theme_bw()+
  theme(
    axis.text.y=element_text(colour="black",size=21,face="plain"),
    axis.title.y=element_text(size =21,face="plain"),
    axis.title.x=element_text(size =21,face="plain"),
    axis.title = element_text(size =24, face="bold",color = 'black'),
    plot.title = element_text(size=24,face="bold",hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line.x = element_line(linetype=1,color="black",size=1.5),
    axis.line.y = element_line(linetype=1,color="black",size=1.5),
    panel.border = element_blank(),
  )+
  ylab("Frequency")+xlab("Fragment size")

# what if I put the two graphs together

dat <- read.table("Hao_fragment.txt",header=T)
dat <- dat %>% mutate(type = as.factor(type))
cdat <- dat %>% group_by(type) %>% summarize(wm_frag = sum(fragment * frequency)) %>% select(type, wm_frag)

ggplot(dat,aes(x = fragment,y = frequency, colour = type))+
  geom_line(size=2,alpha=1) +
  geom_vline(data=cdat, aes(xintercept=wm_frag), linetype="dashed",size=1) +
  geom_text(data = cdat, aes(x = wm_frag, label = wm_frag), 
            y = 0.001, angle = 90, vjust = -1, colour = "black") +
  facet_wrap(~type,scales= "free",ncol=1) +
  #geom_vline(xintercept = c(166),size=1)+
  theme_bw()+
  theme(
    axis.text.y=element_text(colour="black",size=21,face="plain"),
    axis.title.y=element_text(size =21,face="plain"),
    axis.title.x=element_text(size =21,face="plain"),
    axis.title = element_text(size =24, face="bold",color = 'black'),
    plot.title = element_text(size=24,face="bold",hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line.x = element_line(linetype=1,color="black",size=1.5),
    axis.line.y = element_line(linetype=1,color="black",size=1.5),
    panel.border = element_blank(),
  )+
  ylab("Frequency")+xlab("Fragment size")



# anova with discrete Y

library(readxl)
library(dplyr)
HCC_MVI_Sample_Clin <- read_excel("HCC_MVI_Sample_Clin.xlsx")

dat <- HCC_MVI_Sample_Clin
dat <- dat %>% select(MVI, `AFP (μg/L）`, `PIVKA-II (mAU/ml)`, `CA199 (U/ml)`)
colnames(dat) <- c("MVI", "AFP", "PIVKA", "CA199")



# ANOVA univariate analysis, assuming MVI as predictor? 
dat$AFP <- log(as.numeric(dat$AFP))
dat$PIVKA <- log(as.numeric(dat$PIVKA))
dat$CA199 <- log(as.numeric(dat$CA199))
dat_old <- dat
dat$MVI <- ifelse(dat$MVI == "M0", 0, 1)
anova(lm(AFP ~ MVI, data = dat))

anova(lm(PIVKA ~ MVI, data = dat))

anova(lm(CA199 ~ MVI, data = dat))

# multivariate analysis, assuming MVI as response
summary(aov(MVI ~ .,  data = dat))
summary(glm(MVI ~ ., family = binomial(link = "logit"), data = dat))

library(car)
Anova(glm(MVI ~ AFP + PIVKA + CA199, family = binomial(link = "logit"), data = dat))


library(readxl)
library(dplyr)
HCC_MVI_Sample_Clin <- read_excel("HCC_MVI_Sample_Clin.xlsx")

dat <- HCC_MVI_Sample_Clin
dat <- dat %>% select(MVI, `AFP (μg/L）`, `PIVKA-II (mAU/ml)`, `CA199 (U/ml)`)
colnames(dat) <- c("MVI", "AFP", "PIVKA", "CA199")



# ANOVA univariate analysis, assuming MVI as predictor? 
dat$AFP <- log(as.numeric(dat$AFP))
dat$PIVKA <- log(as.numeric(dat$PIVKA))
dat$CA199 <- log(as.numeric(dat$CA199))
dat_old <- dat
dat$MVI <- ifelse(dat$MVI == "M0", 0, ifelse(dat$MVI == "M1", 1, 2))
anova(lm(AFP ~ MVI, data = dat))

anova(lm(PIVKA ~ MVI, data = dat))

anova(lm(CA199 ~ MVI, data = dat))

# multivariate analysis, assuming MVI as response
summary(aov(MVI ~ .,  data = dat))
summary(glm(MVI ~ ., family = binomial(link = "logit"), data = dat))

library(car)
Anova(glm(MVI ~ AFP + PIVKA + CA199, family = binomial(link = "logit"), data = dat))