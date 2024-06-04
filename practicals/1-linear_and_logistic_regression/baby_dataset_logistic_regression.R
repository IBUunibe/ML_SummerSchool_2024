###################################################
## CLASSIFICATION

###################################################
## Logistic regression

setwd("~/Dropbox/Teaching/Courses by Semester/2021 FS/Applied BS II/Topics/data")
babies <- read.table("1-linear_and_logistic_regression/data/baby.dat", header = TRUE)
pairs(babies, pch = 20, col = 2 - babies$Survival, gap = 0.3)

babies.fit <- glm(Survival ~ ., data = babies, 
  family = "binomial")
summary(babies.fit)

# Only weight as explanatory variable
babies.weight.fit <- glm(Survival ~ Weight, data = babies, 
  family = "binomial")
weight <- seq(min(babies$Weight), max(babies$Weight), length.out = 100)
survival.prob <- predict(babies.weight.fit,
  newdata = data.frame(Weight = weight), type = "response")

plot(weight, survival.prob, type = "l", ylim = c(0, 1),
  xlab = "Weight [g]", ylab = "Survival prob.")
points(Survival ~ Weight, data = babies, pch = 20, col = 2 - babies$Survival)

# Misclassification rate
survival.pred <- (predict(babies.weight.fit, type = "response") >= 0.5)
mean(survival.pred != babies$Survival)

