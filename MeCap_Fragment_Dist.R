library(ggplot2)
library(dplyr)

setwd("/Users/yanghao/BerryOncology/甲基化/meCap/Fragment_Methylated_Reads")
dat <- read.table("PFXD117PP13Y21KC001C.fragment.xls",header=T)

cdat <- dat %>% group_by(type) %>% filter(frequency >= max(frequency)) %>% select(type, fragment)

ggplot(dat,aes(x=fragment,y=frequency,group=sample, colour=sample))+
  geom_line(size=2,alpha=1) +
  geom_vline(data=cdat, aes(xintercept=fragment), linetype="dashed",size=1) +
  geom_text(data = cdat, aes(x = fragment, label = type), 
            angle = 90, vjust = -0.2) +
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

cdat