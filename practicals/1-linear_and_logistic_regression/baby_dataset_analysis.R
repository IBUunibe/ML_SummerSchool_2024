set.seed(42)
library(tidyverse) 
library(Rmisc) 
library(tibble) 
library(dplyr)
library(caret)
library(ggplot2)
library(adabag)
library(groupdata2)
library(kableExtra)
library(mlbench)
library(rpart)
library(rpart.plot)
library(patchwork)
library(Hmisc)
library(gridExtra)
library(grid)

# Function for plotting confusion matrices
plot.cm = function(cmtable,angx = 0,angy = 0)
{
  plt <- as.data.frame(cmtable)
  
  plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  
  ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high="#009194", limits = c(0,max(cmtable))) +
    labs(y = "Reference",x = "Prediction") + 
    theme(axis.text.x = element_text(angle = angx, vjust = 0.5, hjust=0.5)) + 
    theme(axis.text.y = element_text(angle = angy, vjust = -0.5, hjust=0.5)) + 
    theme(text = element_text(size = 20)) 
  
}

# here i define a color palette to use as standard when plotting with ggplot
# this is purely for aesthetic reasons

cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

scale_colour_discrete <- function(...) 
{ scale_colour_manual(..., values = cbp2)
}


#------
df <- read.csv("1-linear_and_logistic_regression/data/baby.dat", sep="")

df$Survival = as.factor(df$Survival)

# I rename the response variable to "labels"
df = dplyr::rename(df,labels = Survival)
levels(df$labels) = c("not surviving","surviving")

#------
# uncomment this to load a different data-set

#data(iris)
#df = iris
#df = dplyr::rename(df,labels = Species)


#------
# uncomment this to load a different data-set

# Wisconsin Breast Cancer Database
# data(BreastCancer)
# head(BreastCancer)
# levels(BreastCancer$Class)
# sum(is.na(BreastCancer))
# df = na.omit(BreastCancer[,-1])
# head(df)
# df = dplyr::rename(df,labels = Class)
# head(df)



#-------
# Split the data into test and training data

trn_indx <- createDataPartition(df$labels , 
                                p = .7, 
                                list = FALSE,times = 1) %>%  as.numeric()

tst_indx <- which(!(seq_len(nrow(df)) %in% trn_indx))

train = df[trn_indx,]
test = df[tst_indx,]

table(train$labels)
table(test$labels)

#------
# uncomment this for balancing the data set:

#train = train  %>%
# balance(cat_col = "labels", size = "min")
# table(train$labels)

# test = test  %>%
#  balance(cat_col = "labels", size = "max")


#-------
# Center and scale the data:

(preproc <- preProcess(train, method = c("center","scale"))) # note that factors are not transformed
pp_train <- predict(preproc, train) # centering and Scaling train set based on train data

(preproc <- preProcess(test, method = c("center","scale"))) # note that factors are not transformed
pp_test <- predict(preproc, test) # centering and Scaling test set based on train data

#-----
# plot the data, this only works for the baby data set

#ggplot() + 
#  geom_point(data = pp_train,
#             aes(x = Weight,y = Age,fill = labels),shape=21)  +
#  geom_point(data = pp_test,
#             aes(x = Weight,y = Age,fill = labels),col = "slategray",shape = 24) + 
#  theme_minimal()



#------
# Fit a decision tree using the package rpart


mod = rpart(labels~.,minsplit=10,dat=pp_train)

# vanilla fitting
rpart.plot(mod,type = 5)

# calcualte misclassification rate:
pred = predict(mod,pp_test,type="class")
pred.in.sample = predict(mod,pp_train,type="class")

mis.class.full = mean(pred != pp_test$labels)
mis.class.full.in.sample =mean(pred.in.sample != pp_train$labels)

mis.class.full
mis.class.full.in.sample


# cross validation of the performance
plotcp(mod)

# optimal hyperparameter
cp = mod$cptable[which.min(mod$cptable[,4]),1]
cp
# prune the tree using the manually selected hyperparamater cp
mod = prune(mod,cp=cp)

# pruned tree
rpart.plot(mod,type = 5,clip.right.labs = FALSE, branch = .3, under = TRUE)

# calcualte misclassification rate:
pred = predict(mod,pp_test,type="class")
pred.in.sample = predict(mod,pp_train,type="class")

mis.class.pruned = mean(pred != pp_test$labels)
mis.class.pruned.in.sample =mean(pred.in.sample != pp_train$labels)

df.miss.class = data.frame(
  model = c("full","full","pruned","pruned"),
  data.set = c("test","train","test","train"),
  error = c(mis.class.full,mis.class.full.in.sample,
            mis.class.pruned,mis.class.pruned.in.sample)
)

plot1 = ggplot(dat = df.miss.class) + 
  geom_bar(stat="identity",
           aes(y = error,x = model,fill = data.set),
           position=position_dodge()) + 
  theme_minimal() + 
  ylim(0,0.3)

show(plot1)

# plot a confusion Matrix
(cm <- confusionMatrix(pred, pp_test$labels))
plot.cm(cm$table)
cm$byClass
# Summarize perfromance measures
cm.dat.dt = data.frame(RPart = cm$byClass)
cm.dat.dt %>%
  kbl() %>%
  kable_styling()

# the same exercise in the caret framework
set.seed(10)
mod2 = caret::train(labels~ ., 
                        data=pp_train,
                        method = 'rpart')

pred = predict(mod2,pp_test,type="raw")
pred.in.sample = predict(mod2,pp_train,type="raw")

mean(pred != pp_test$labels)
mean(pred.in.sample != pp_train$labels)



(cm <- confusionMatrix(pred, pp_test$labels))
plot.cm(cm$table)


#------
# The same exercise with Random Forests

set.seed(10)
rf_train = caret::train(labels~ ., 
                        data=pp_train,
                        method = 'parRF')


plot(rf_train)
rf_train$results[rownames(rf_train$bestTune),]
pred <- predict(rf_train, pp_test)
pred

(cm <- confusionMatrix(pred, pp_test$labels))
plot.cm(cm$table,angx = 0,angy=0)

roc_imp2 <- varImp(rf_train, scale = FALSE)
plot(roc_imp2,top=5,xlim=c(0,max(roc_imp2$importance)))


tunegrid <- expand.grid(
  mtry = c(1:6)
)

set.seed(10)
trcontrol <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3, 
                          search = "grid") 

rf_train = caret::train(labels~ ., 
                        data=pp_train,
                        trControl = trcontrol,
                        tuneGrid = tunegrid,
                        method = 'parRF')

plot(rf_train)

pred <- predict(rf_train, pp_test)


(cm <- confusionMatrix(pred, pp_test$labels))
plot.cm(cm$table,angx = 0,angy=0)

roc_imp2 <- varImp(rf_train, scale = FALSE)
plot(roc_imp2,top=5,xlim=c(0,max(roc_imp2$importance)))



cm.dat.rf = data.frame(RandomForest = cm$byClass)
cm.dat.rf %>%
  kbl() %>%
  kable_styling()

#------
# The same exercise with XGBoost

set.seed(10)
tunegrid <- expand.grid(nrounds = seq(1,30,by=2),
                        max_depth = c(1,5,10),
                        eta = seq(0.01,0.5,length.out = 5),
                        gamma = 1,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)



trcontrol <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 5, 
                          search = "random",
                          savePredictions = TRUE) 

xgb_train = caret::train(labels~ ., 
                         data=pp_train,
                         trControl = trcontrol,
                         tuneGrid = tunegrid,
                         method = "xgbTree")

plot(xgb_train)


xgb_train$bestTune

xgb_train$results[rownames(xgb_train$bestTune),]
pred <- predict(xgb_train, pp_test)
(cm <- confusionMatrix(pred, pp_test$labels))
plot.cm(cm$table,angx = 0,angy = 0)


roc_imp2 <- varImp(xgb_train, scale = FALSE)
plot(roc_imp2,top=5,xlim=c(0,max(roc_imp2$importance+0.02)))
pred <- predict(xgb_train, pp_train)


plot(roc_imp2,top=5,xlim=c(0,max(roc_imp2$importance+0.02)))

# summarize results

cm.dat.xg = data.frame(XGBoost = cm$byClass)
cm.dat.xg %>%
  kbl() %>%
  kable_styling()

cm.dat = cbind((cm.dat.dt),(cm.dat.rf),(cm.dat.xg))
t(cm.dat) %>%
  kbl() %>%
  kable_styling()

#-----
# Here we take a deeper look at 
# posterior probabilities of the different classifiers


get.ratio = function(pred,labs,p1 = 0,p2 = 0.5)
{
  ratio=NA
  ind = (pred[,2] >= p1 & pred[,2] < p2 )
  if(length(ind) >= 1)
    ratio = sum(labs[ind] == 1)/(sum(ind))
  return(c(ratio,length(labs[ind])))
}

plotProb = function(pred)
{
  inc = 0.1
  p2 = seq(0.1,1,by=inc)
  n = length(p2)
  ratios = numeric(n*2)
  dim(ratios)=c(n,2)
  cis = ratios
  
  for(i in 1:n)
  {
    ratios[i,] = get.ratio(pred,pp_test$labels,p2[i]-inc,p2[i])
    if(!is.nan(ratios[i,1]))
      cis[i,] = binconf(x=ratios[i,1]*ratios[i,2], n=ratios[i,2], alpha=.05)[c(2,3)]
    
  }
  prob.dat = data.frame(probs = p2 - inc/2,ratios = ratios[,1],
                        observations = ratios[,2],
                        lower = cis[,1],upper = cis[,2])
  
  
  plot1 = ggplot(prob.dat[complete.cases(prob.dat), ],aes(x = probs,y = ratios)) +
    geom_line() + 
    geom_abline(slope=1,intercept=0,lty=2) + 
    geom_point(aes(size = observations)) + 
    xlab(paste("predicted probability")) + 
    ylab(paste("observed probability")) + 
    theme_classic()  +
    ylim(c(0,1)) + 
    xlim(c(0,1))    +
    geom_errorbar(aes(ymin=lower,ymax=upper), colour="black", width=.1) 
  
  dens.df = data.frame(pred.prob = pred[,2])
  
  plot2 = ggplot()+ 
    geom_density(dat = dens.df,aes(x = pred.prob),bw = 0.05) + 
    geom_rug() +
    xlab(paste("predicted probability")) +
    theme_classic() + 
    xlim(c(0,1))

  #plot.final = plot1 + plot2 + plot_layout(ncol = 1)
  #show(plot.final)
  return(plot1)
}

#pp_test$labels=as.numeric(pp_test$labels)-1

pp_test$labels = as.factor(as.numeric(pp_test$labels)-1)
pp_test$labels
pred.rPart <- predict(mod, pp_test,type="prob")
pred.RF <- predict(rf_train, pp_test,type="prob")
pred.XGB <- predict(xgb_train, pp_test,type="prob")


p1 = plotProb(pred.rPart)
p2 = plotProb(pred.RF)
p3 = plotProb(pred.XGB)

grid.arrange(p1, p2, p3, ncol=3)


