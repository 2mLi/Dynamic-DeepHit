# this script is some of my functions that make works smoother

# log1(): a function that log-transform the data
# but every data point was added by 0.00001 (modifi-able)
# to circumvent problems of zero (0)
# the function also accepts "base" argument so log_10 or log_2 is possible

log1 <- function(x, a = 0.00001, base = exp(1)){
  return(log(x + 0.00001, base = base))
}

toSurv <- function(dat, idvar, timevar, newtimevar){
  # convert dat to dat.id format which is acceptable to coxph
  ID <- unique(dat[, idvar]) # patient list
  this_patient <- dat[dat[, idvar] == ID[1], ]
  output <- this_patient[1, ]
  output[, newtimevar] <- max(this_patient[, timevar])
  
  for (i in 2:length(ID)){
    this_patient <- dat[dat[, idvar] == ID[i], ]
    to_output <- this_patient[1, ]
    to_output[, newtimevar] <- max(this_patient[, timevar])
    output <- rbind(output, to_output)
  }
  return(output)
}

toDummy <- function(x){
  # this function converts x into a sort of dummy
  # for example, x <- c(x, x, y, z) will become c(1, 1, 2, 3)
  
  unique_x <- unique(x) # unique levels in x
  l <- length(x)
  rep <- 1:length(unique_x)
  x2 <- rep(NA, l)
  for (i in 1:l){
    x2[i] <- rep[unique_x == x[i]]
  }
  return(x2)
}


zhuanhua.Shuzi <- function(x){
  # convert x from chinese to numeric characters
  if (x == "一" | x == "壹"){
    return(1)
  }else if (x == "二" | x == "贰"){
    return(2)
  }else if (x == "三" | x == "叁"){
    return(3)
  }else if (x == "四" | x == "肆"){
    return(4)
  }else if (x == "五" | x == "伍"){
    return(5)
  }else if (x == "六" | x == "陆"){
    return(6)
  }else if (x == "七" | x == "柒"){
    return(7)
  }else if (x == "八" | x == "捌"){
    return(8)
  }else if (x == "九" | x == "玖"){
    return(9)
  }else if (x == "零" | x == "拾"){
    return(0)
  }
}

last <- function(x){
  return(x[length(x)])
}

within <- function(x, p1, p2){
  # returns TRUE if at least one element of x lies between p1 and p2
  p11 <- x >= p1
  p22 <- x <= p2
  y <- p11 & p22
  z <- table(y)
  # if at least one element of x is within [p1, p2]: there would be at least one TRUE
  if (length(z) == 2){
    return(TRUE)
  }else if (length(z) == 1 & y[1] == TRUE){
    return(TRUE)
  }else{
    return(FALSE)
  }
  
}