library(readr)
library(readsparse)
library(readr)
library(dplyr)
library(stargazer)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

#to read sparse data

train <-read.sparse("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_train.data")

train_y <- as.matrix(read_csv("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_train.labels", 
                              col_names = FALSE))
test <-read.sparse("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_valid.data")
test_y <- as.matrix(read_csv("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_valid.labels", 
                             col_names = FALSE))

# train mean and sd
x_mean=as.numeric(colMeans(as.matrix(train$X)))
x_sd =as.numeric(apply(as.matrix(train$X),2,sd))

# Rename labels
train_y[train_y== -1] = 0
test_y[test_y== -1] = 0
#setup data
train$x <- as.matrix(cbind(rep(1,nrow(train$X)),scale(train$X,x_mean,x_sd)))
test$x <- as.matrix(cbind(rep(1,nrow(test$X)),scale(test$X,x_mean,x_sd)))

X = rbind(train$x, test$x)
X = X[, colSums(is.na(X)) == 0]
data_train = list(y=train_y,x=as.matrix(X[1:300,]));str(data_train)
data_test= list(y=test_y,x=as.matrix(X[301:600,]));str(data_test)

logistic = function(n_iter=100,
                    data_train,
                    data_test,
                    eta=1/nrow(data_train$y),
                    lambda,
                    w_init = as.matrix(rep(0, ncol(data_train$x))))
{
  w_new = w_init
  n = 1
  my.list = list()
  train_miss = array()
  iteration = array()
  y_train = data_train$y
  y_test = data_test$y
  # Updating w
  
  while (n <= n_iter)
  {
    w = w_new
    pred = data_train$x %*% w
    v = as.matrix(data_train$y - (1+exp(-pred))^(-1))
    #updating coefficients
    w_new_theta = w + (1/eta)*t(data_train$x)%*%v
    w_new = ifelse(abs(w_new_theta) > lambda,w_new_theta,0)
    n = n + 1
  } 
  
  
  my.list$num_feature = length(which(w_new!=0))
  ## Train Data
  link_train = data_train$x %*% w_new
  roc_train = roc(as.numeric(y_train), as.numeric(link_train))
  threshold_train = as.numeric(coords(roc_train, "best", ret = "threshold",drop=TRUE)[1])
  y_hat_train = as.factor(ifelse(link_train > threshold_train, 1, 0))
  levels(y_hat_train) = c("0", "1")
  
  cm_train = confusionMatrix(as.factor(y_train), as.factor(y_hat_train))
  my.list$train_miss = as.numeric(1 - cm_train$byClass['Balanced Accuracy'])
  ## Test Data
  link_test = data_test$x %*% w_new
  # prob_test = exp(link_test)/(1+exp(link_test))
  roc_test = roc(as.numeric(y_test), as.numeric(link_test))
  threshold_test = as.numeric(coords(roc_test, "best", ret = "threshold",drop=TRUE)[1])
  y_hat_test = as.factor(ifelse(link_test > threshold_test, 1, 0))
  levels(y_hat_test) = c("0", "1")
  cm_test = confusionMatrix(as.factor(y_test), as.factor(y_hat_test))
  my.list$test_miss = as.numeric(1 - cm_test$byClass['Balanced Accuracy'])
  return(my.list)
}

final_logistic = function(lambda1)
{
  r1 = logistic(100, data_train, data_test, lambda=lambda1)
  out = data.frame(lambda1,r1$num_feature,r1$train_miss,r1$test_miss)
  colnames(out) = c("lambda","Feature","Miss.Train","Miss.Test")
  rownames(out) = NULL
  return(out)
}

lambda_all = c(9912,7313,6155,5040.268,4213)

my.cluster <- makeCluster(10)
registerDoParallel(my.cluster)
clusterExport(my.cluster,c("final_logistic","logistic","data_train","data_test"),envir = .GlobalEnv)
invisible(clusterEvalQ(my.cluster,
                       {library(dplyr)
                         library(stargazer)
                         library(caret)
                         library(pROC)
                       }))
u=parSapply(my.cluster,lambda_all,final_logistic)
t(u)
stopCluster(my.cluster)
D=data.frame(t(u))
names(D)
D['Miss.Train']
stopCluster(my.cluster)
#train vs test missplot
plot = ggplot(D,aes(x= as.numeric(unlist(D['Feature']))))+
  geom_line(aes(y = as.numeric(unlist(D['Miss.Train'])),color = "Training"))+
  geom_line(aes(y = as.numeric(unlist(D['Miss.Test'])),color = "Testing"))+
  ylim(0,0.2) + ylab('Misclassification Error')+ xlab('Number of Feature')



# Missclassification Error Plot
Miss_plot = function(n_iter,lambda,eta,data=data_train)
{
  w_new = as.matrix(rep(0, ncol(data$x)))
  n = 1
  train_miss = array()
  iteration = array()
  y_train = data$y
  invisible(while(n <= n_iter){
    w = w_new
    pred = data$x %*% w
    v = as.matrix(data$y - (1+exp(-pred))^(-1))
    #updating coefficients
    w_new_theta = w + (1/eta)*t(data$x)%*%v
    w_new = ifelse(abs(w_new_theta) > lambda,w_new_theta,0)
    link_train = data$x %*% w_new
    roc_train = roc(as.numeric(y_train), as.numeric(link_train))
    threshold_train = as.numeric(coords(roc_train, "best", ret = "threshold",drop=TRUE)[1])
    y_hat_train = as.factor(ifelse(link_train > threshold_train, 1, 0))
    levels(y_hat_train) = c("0", "1")
    cm_train = confusionMatrix(as.factor(y_train), as.factor(y_hat_train))
    train_miss[n] = as.numeric(1 - cm_train$byClass['Balanced Accuracy'])
    #keeping iteration number
    iteration[n] = n
    #updating iteration
    n = n + 1
  } )
  data=data.frame(iteration,train_miss)
  miss.plot = ggplot(data, aes(x = iteration)) + 
    geom_line(aes(y = train_miss))+
    ylab('Missclassification Errors')+xlab('Iteration')+
    annotate("text", x=75, y=0.02, label= "Feature = 100",
             col="blue", size=5)
  return(miss.plot)
}
Miss_plot(100,lambda=6155,eta=1/nrow(data_train$y),data_train)

# Roc Plot
Roc_plot = function(n_iter,lambda,eta,train,test)
{
  
  w_new = as.matrix(rep(0, ncol(data_train$x)))
  n = 1
  my.list = list()
  y_train = train$y
  y_test = test$y
  while (n <= n_iter)
  {
    w = w_new
    pred = train$x %*% w
    v = as.matrix(train$y - (1+exp(-pred))^(-1))
    #updating coefficients
    w_new_theta = w + (1/eta)*t(train$x)%*%v
    w_new = ifelse(abs(w_new_theta) > lambda,w_new_theta,0)
    n = n + 1
  } 
  link_train = train$x %*% w_new
  roc_train = roc(as.numeric(y_train), as.numeric(link_train))
  link_test = data_test$x %*% w_new
  roc_test = roc(as.numeric(y_test), as.numeric(link_test))
  plot = ggroc(list(Train = roc_train, Test = roc_test ))+
    geom_abline(slope=1,intercept = 1,color = "pink")+
    annotate("text", x=0.125, y=0.25, label= "Feature = 500",
             col="blue", size=5)
  
  return(plot)
}

Roc_plot(100,lambda=4213,eta=1/nrow(data_train$y),data_train,data_test)

# 
# job({p11=uniroot(function(x) {500-logistic(100, data_train, data_test, lambda =x)$num_feature},
#                 interval = c(3000,10000))$root})
# job({p22=uniroot(function(x) {300-logistic(100, data_train, data_test, lambda =x)$num_feature},
#                 interval = c(3000,10000))$root})
# job({p33=uniroot(function(x) {100-logistic(100, data_train, data_test, lambda =x)$num_feature},
#                 interval = c(3000,10000))$root})
# job({p44=uniroot(function(x) {30-logistic(100, data_train, data_test, lambda =x)$num_feature},
#                 interval = c(3000,10000))$root})
# job({p55=uniroot(function(x) {10-logistic(100, data_train, data_test, lambda =x)$num_feature},
#                 interval = c(3000,10000))$root})
