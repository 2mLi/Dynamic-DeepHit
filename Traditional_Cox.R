library(survival)
library(survminer)
library(survcomp)
library(plyr)
source("my_functions.R", encoding = "UTF-8")



Train <- read.csv("data/precar_Train_WGS_plus.csv")

Test <- read.csv("data/precar_Test_WGS_plus.csv")

# variables used by DDH
bin_list <- c("Gender", "CNV_score")
cont_list <- c("Afp", "Age", "Alt", "Alb", "Plt", "Tb", "Inr", "NF_CT", "Fragment_CT", "motif_CT", "Comb_CT", "score")
all_list <- c(bin_list, cont_list)
log_transform <- c("Afp", "Alt", "Alb", "Plt", "Tb", "Inr", "Comb_CT", "score")

Train <- Train[, c(bin_list, cont_list, "Time", "Times", "ID", "Status")]

Test <- Test[, c(bin_list, cont_list, "Time", "Times", "ID", "Status")]

# log-transform
for (var in log_transform){
  Train[, var] <- log1(Train[, var], base = 10)
  Test[, var] <- log1(Test[, var], base = 10)
}

# to cox data
Train_id <- toSurv(Train, "ID", "Time", "Times")
Test_id <- toSurv(Test, "ID", "Time", "Times")

# fit coxph

# univariate

UniCox<-function(x){
  FML<-as.formula(paste0("Surv(Times, Status) ~ ",x))
  dat <- Train_id
  Cox<-coxph(FML,data = dat) 
  Sum<-summary(Cox)
  CI<-paste0(round(Sum$conf.int[,3:4],2),collapse = "-") 
  Pvalue<-round(Sum$coefficients[,5],4)
  HR<-round(Sum$coefficients[,2],2)
  Unicox<-data.frame("Characteristics"=x,
                     "Hazard Ratio"=HR,
                     "CI95"=CI,
                     "P value"=Pvalue)
  return(Unicox)
}

varNames<- as.list(all_list)
dat <- Train_id
UniVar<-lapply(varNames, UniCox)
# extract p values manually
UniVar_res <- ldply(UniVar, function(x){return(x$P.value <= 0.05 / length(UniVar))}) # result of univariate analysis
selected_predictors <- all_list[UniVar_res[, 1]]
selected_predictors
# [1] "Afp"   "Age"   "Alb"   "Plt"   "Inr"   "NF_CT" "score"


# multivariate

mod <- coxph(Surv(Times, Status) ~ Afp + Age + score, 
             data = Train_id)

# predict (Train)
tr_risk <- predict(mod, Train_id, type="risk")
Train_id$risk <- tr_risk

#选择最优阈值
res.cut <- surv_cutpoint(Train_id, time = "Time", event = "Status",variables = c("risk"))
cutoff <- res.cut$risk$estimate

Train_id$Group <- ifelse(Train_id$risk >= cutoff,"High","Low")

sum.surv <- summary(mod)
c_index_se <- sum.surv$concordance
c_index <- c_index_se[1]
c_index.ci_low <- c_index - c_index_se[2]
c_index.ci_upp <- c_index + c_index_se[2]

ggsurvplot(mod, # 创建的拟合对象
           data = Train_id,  # 指定变量数据来源
           conf.int = TRUE, # 显示置信区间
           pval = TRUE, # 添加P值
           fun = "event",
           pval.size = 5,
           title = paste0("Derivation Cohort(cutpoint: ",round(cutoff,3),", ","C-index: ",round(c_index,3),")"),
           #surv.median.line = "hv",  # 添加中位生存时间线
           risk.table = TRUE, # 添加风险表
           xlab = "Follow up time(Month)", # 指定x轴标签
           ylab = "Cumulative risk of hepatocellular carcinoma",
           legend = c(0.15,0.8), # 指定图例位置
           legend.title = "aMAP-plus Score", # 设置图例标题，这里设置不显示标题，用空格替代
           legend.labs = c("High-risk group", "Low-risk group"), # 指定图例分组标签
           break.x.by = 3,
           xlim = c(0,36),
           #ylim = c(0,0.4)
)  # 设置x轴刻度间距


# whatever, on test set... 
te_risk <- predict(mod, Test_id, type="risk")
Test_id$risk <- te_risk
Test_id$Group <- ifelse(Test_id$risk >= cutoff,"High","Low")
idx <- !is.na(te_risk)
c_index_te <- concordance.index(x = te_risk[idx], surv.time = Test_id$Times[idx], 
                                surv.event = Test_id$Status[idx])$c.index

# with baseline data, 