## Load packages--------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(randomForest)
library(base)

## Set the Working Directory
setwd("C:/Users/Fabio/Documents/Data Science/Master in Data Science and Society/Programming in R/Group Project/Code")

## Load data
student_portuguese_language<-read.csv("input/student-por.csv", stringsAsFactors = FALSE, sep = ";")
student_portuguese_language <- read.delim("input/student-por.csv",sep=";",
                                          header=TRUE,stringsAsFactors = FALSE)

datasetwithport <- student_portuguese_language


### Pre-Processing -------------------------------------------------------

##Rename the dataset.
new_student <- student_portuguese_language
View(new_student)
## Check for missing values.
na_check <- sapply(new_student, function(x) sum(is.na(x)))
na_check

##Drop the least important columns.
new_student <- select(new_student,-(4:5),-(9:12),-(16:20),-(23:29))
new_student
ncol(new_student)
View(new_student)
##Rename some columns.
names(new_student)[4] <-"parent_cohabitation_status"
names(new_student)[5] <-"mother_education"
names(new_student)[6] <-"father_education"
names(new_student)[10] <-"higher_education"
names(new_student)[13] <-"first_period_grade"
names(new_student)[14] <-"second_period_grade"
names(new_student)[15] <-"final_grade"

##Overview of the dataset.
head(new_student)
tail(new_student)
str(new_student)
summary(new_student)

##Plot correlations between grades' columns(highly-correlated).
my_cols <- c("red","green","black")
pairs(new_student[,13:15],col = my_cols,lower.panel = NULL)

##Boxplot of final_grade per sex.
sex_grade <- ggplot(new_student,aes(x = sex,y = final_grade,fill=sex)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Final_Grade per Sex",x="Sex (Female,Male)",
       y = "Final_Grade (0-20)",fill = "Sex")
sex_grade

##Boxplot of final_grade per school.
school_grade <- ggplot(new_student,aes(x = school,y = final_grade,fill=school)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Final_Grade per School",
       x="School (Gabriel Pereira,Mousinho da Silveira)",
       y = "Final_Grade (0-20)",fill = "School")
school_grade

##Boxplot of final_grade per parent_cohabitation_status.
status_grade <- ggplot(new_student,aes(x = parent_cohabitation_status,
                                       y = final_grade,fill=parent_cohabitation_status)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Final_Grade per Status",
       x="Status (Apart,Living Together)",
       y = "Final_Grade (0-20)",fill = "Parent_Cohabitation_Status")
status_grade

##Boxplot of final_grade per higher_education.
higher_education_grade <- ggplot(new_student,aes(x = higher_education,
                                                 y = final_grade,fill=higher_education)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Final_Grade per Higher_Education",
       x="Higher_Education (No,Yes)",
       y = "Final_Grade (0-20)",fill = "Higher_Education")
higher_education_grade

##Boxplot of final_grade per internet.
internet_grade <- ggplot(new_student,aes(x = internet,y = final_grade,fill=internet)) + 
  geom_boxplot() + 
  labs(title="Boxplot of Final_Grade per Internet",
       x="Internet (No,Yes)",
       y = "Final_Grade (0-20)",fill = "Internet")
internet_grade

##Change the number of target class clusters to make better predictions.
new_student$final_grade[new_student$final_grade<=5] <- 1
new_student$final_grade[new_student$final_grade>=6 & new_student$final_grade<=10] <- 2
new_student$final_grade[new_student$final_grade>=11 & new_student$final_grade<=15] <- 3
new_student$final_grade[new_student$final_grade>=16 & new_student$final_grade<=20] <- 4
str(new_student)
##Final form of student dataset.
student_df <- new_student
student_df
View(student_df)
str(student_df)
##creating factors and integers from dataset
cols <- c("school", "sex", "parent_cohabitation_status", "higher_education", "internet", "final_grade")
student_df[cols] <- lapply(student_df[cols], factor) 
sapply(student_df, class)
View(student_df)
str(student_df)
## Creating a data partition for the models
set.seed(1)
trn_index = createDataPartition(y = student_df$final_grade, p = 0.80, list = FALSE)
trn_student = student_df[trn_index, ]
tst_student = student_df[-trn_index, ]

trn_student

### KNN Model -------------------------------------------------------
set.seed(1)
grade_knn = train(final_grade ~.-(first_period_grade + second_period_grade),
                  method = "knn", data = trn_student,
                  trControl = trainControl(method = 'cv', number = 5), 
                  preProcess = c("center", "scale"), 
                  tuneLength = 5)

grade_knn$results

plot_knn_results <- function(fit_knn) {
  ggplot(fit_knn$results, aes(x = k, y = Accuracy)) +
    geom_bar(stat = "identity") +
    scale_x_discrete("value of k", limits = fit_knn$results$k) +
    scale_y_continuous("accuracy")}

plot_knn_results(grade_knn)

predict_knn = predict(grade_knn, tst_student)

tst_student$final_grade <- factor(tst_student$final_grade)

knn_confM <- confusionMatrix(predict_knn, tst_student$final_grade)

knn_confM

knn_confM$overall #Accuracy of 0.697674419


### Logistic Regression Model -------------------------------------------------------


##testing the other dataset (Portugese language)
##Drop the least important columns. Don't run it twice as it will edit the dataset again.
datasetwithport <- select(datasetwithport,-(4:5),-(9:12),-(16:20),-(23:29))

##Renaming of columns
names(datasetwithport)[4] <-"parent_cohabitation_status"
names(datasetwithport)[5] <-"mother_education"
names(datasetwithport)[6] <-"father_education"
names(datasetwithport)[10] <-"higher_education"
names(datasetwithport)[13] <-"first_period_grade"
names(datasetwithport)[14] <-"second_period_grade"
names(datasetwithport)[15] <-"final_grade"

##creating the one vs all datasets
datawithpositive1 <- datasetwithport
datawithpositive2 <- datasetwithport
datawithpositive3 <- datasetwithport
datawithpositive4 <- datasetwithport

datawithpositive1$final_grade[datasetwithport$final_grade<=5] <- 1
datawithpositive1$final_grade[datasetwithport$final_grade>=6] <- 2

datawithpositive2$final_grade[datasetwithport$final_grade>=6 & datasetwithport$final_grade <= 10] <- 1
datawithpositive2$final_grade[datasetwithport$final_grade<=5 | datasetwithport$final_grade >= 11] <- 2

datawithpositive3$final_grade[datasetwithport$final_grade>=11 & datasetwithport$final_grade <= 15] <- 1
datawithpositive3$final_grade[datasetwithport$final_grade<=10 | datasetwithport$final_grade >= 16] <- 2

datawithpositive4$final_grade[datasetwithport$final_grade>=16 & datasetwithport$final_grade <= 20] <- 1
datawithpositive4$final_grade[datasetwithport$final_grade<=15] <- 2

##making factors out of the columns
cols <- c("school", "sex", "parent_cohabitation_status", "higher_education", "internet", "final_grade")
datawithpositive1[cols] <- lapply(datawithpositive1[cols], factor) 
sapply(datawithpositive1, class)
datawithpositive2[cols] <- lapply(datawithpositive2[cols], factor) 
sapply(datawithpositive2, class)
datawithpositive3[cols] <- lapply(datawithpositive3[cols], factor) 
sapply(datawithpositive3, class)
datawithpositive4[cols] <- lapply(datawithpositive4[cols], factor) 
sapply(datawithpositive4, class)

## Creating a data partition for the models, model 1 first
set.seed(1)
trn_index1 = createDataPartition(y = datawithpositive1$final_grade, p = 0.70, list = FALSE)
trn_student1 = datawithpositive1[trn_index1, ]
tst_student1 = datawithpositive1[-trn_index1, ]

# ## Adding weights to balance the dataset, has been left out at the moment
# sum(trn_student$final_grade == 1)
# sum(tst_student$final_grade == 1)
# 
# trn_student <- trn_student %>%
#   mutate(weights = case_when(trn_student$final_grade == 1 ~ 0.65,
#                                 trn_student$final_grade == 2 ~ 0.35))
# 
# tst_student <- tst_student %>%
#   mutate(weights = case_when(tst_student$final_grade == 1 ~ 0.65,
#                              tst_student$final_grade == 2 ~ 0.35))


## creating model 1
student_lgr1 = train(final_grade ~ . - (first_period_grade + second_period_grade), method = "glm",
                     family = binomial(link = "logit"), data = trn_student1,
                     trControl = trainControl(method = 'cv', number = 5))

student_lgr1
summary(student_lgr1)

## predicting the outcomes
predicted_outcomes1 <- predict(student_lgr1, tst_student1)
accuracy <- sum(predicted_outcomes1 == tst_student1$final_grade) /
  length(tst_student1$final_grade)
accuracy ##[1] 0.9793814

## creating model 2
set.seed(1)
trn_index2 = createDataPartition(y = datawithpositive2$final_grade, p = 0.70, list = FALSE)
trn_student2 = datawithpositive2[trn_index1, ]
tst_student2 = datawithpositive2[-trn_index1, ]

student_lgr2 = train(final_grade ~ . - (first_period_grade + second_period_grade), method = "glm",
                     family = binomial(link = "logit"), data = trn_student2,
                     trControl = trainControl(method = 'cv', number = 5))

student_lgr2
summary(student_lgr2)

## predicting the outcomes
predicted_outcomes2 <- predict(student_lgr2, tst_student2)
accuracy <- sum(predicted_outcomes2 == tst_student2$final_grade) /
  length(tst_student2$final_grade)
accuracy ##[1] 0.7938144

## creating model 3
set.seed(1)
trn_index3 = createDataPartition(y = datawithpositive3$final_grade, p = 0.70, list = FALSE)
trn_student3 = datawithpositive3[trn_index1, ]
tst_student3 = datawithpositive3[-trn_index1, ]

student_lgr3 = train(final_grade ~ . - (first_period_grade + second_period_grade), method = "glm",
                     family = binomial(link = "logit"), data = trn_student3,
                     trControl = trainControl(method = 'cv', number = 5))

student_lgr3
summary(student_lgr3)

## predicting the outcomes
predicted_outcomes3 <- predict(student_lgr3, tst_student3)
accuracy <- sum(predicted_outcomes3 == tst_student3$final_grade) /
  length(tst_student3$final_grade)
accuracy ##[1] 0.7010309

## creating model 4
set.seed(1)
trn_index4 = createDataPartition(y = datawithpositive4$final_grade, p = 0.70, list = FALSE)
trn_student4 = datawithpositive4[trn_index1, ]
tst_student4 = datawithpositive4[-trn_index1, ]

student_lgr4 = train(final_grade ~ . - (first_period_grade + second_period_grade), method = "glm",
                     family = binomial(link = "logit"), data = trn_student4,
                     trControl = trainControl(method = 'cv', number = 5))

student_lgr4
summary(student_lgr4)

## predicting the outcomes
predicted_outcomes4 <- predict(student_lgr4, tst_student4)
accuracy <- sum(predicted_outcomes4 == tst_student4$final_grade) /
  length(tst_student4$final_grade)
accuracy ##[1] 0.876288



## Calculating the probability of predicting a certain grade and extracting a predicted class from it.
## This code needs to loop over the probablities for each index and get the highest number.

probabilitiesmodel1 <- predict(student_lgr1, tst_student1, type = 'prob')
probabilitiesmodel2 <- predict(student_lgr2, tst_student2, type = 'prob')
probabilitiesmodel3 <- predict(student_lgr3, tst_student3, type = 'prob')
probabilitiesmodel4 <- predict(student_lgr4, tst_student4, type = 'prob')

#Store the probability of predicting 1 for each model
probs_for_one <- cbind(probabilitiesmodel1$`1`,probabilitiesmodel2$`1`,probabilitiesmodel3$`1`,probabilitiesmodel4$`1`)

# find and store the max probability over all models
max_of_probs_for_one <- apply(probs_for_one, 1, max)

#convert the probabilites of each model into predictions based on the max probability over all models
final_predictions <- data.frame(probs_for_one)
final_predictions$X1[probs_for_one[,1] == max_of_probs_for_one] <- 1
final_predictions$X1[probs_for_one[,1] != max_of_probs_for_one] <- 2
final_predictions$X2[probs_for_one[,2] == max_of_probs_for_one] <- 1
final_predictions$X2[probs_for_one[,2] != max_of_probs_for_one] <- 2
final_predictions$X3[probs_for_one[,3] == max_of_probs_for_one] <- 1
final_predictions$X3[probs_for_one[,3] != max_of_probs_for_one] <- 2
final_predictions$X4[probs_for_one[,4] == max_of_probs_for_one] <- 1
final_predictions$X4[probs_for_one[,4] != max_of_probs_for_one] <- 2

# create full tst set
all_tst <- cbind(tst_student1$final_grade,tst_student2$final_grade,tst_student3$final_grade,tst_student4$final_grade)

#calculate the final accuracy
accuracy_log_reg <- sum(final_predictions == all_tst) / length(all_tst)
accuracy_log_reg ## [1] 0.8402062

### Random Forest Model -------------------------------------------------------

datasetwithport <- student_portuguese_language

##testing the other dataset (Portugese language)
##Drop the least important columns.
datasetwithport <- select(datasetwithport,-(4:5),-(9:12),-(16:20),-(23:29))

##Rename some columns.
names(datasetwithport)[4] <-"parent_cohabitation_status"
names(datasetwithport)[5] <-"mother_education"
names(datasetwithport)[6] <-"father_education"
names(datasetwithport)[10] <-"higher_education"
names(datasetwithport)[13] <-"first_period_grade"
names(datasetwithport)[14] <-"second_period_grade"
names(datasetwithport)[15] <-"final_grade"

datasetwithport$final_grade[datasetwithport$final_grade<=5] <- 1
datasetwithport$final_grade[datasetwithport$final_grade>=6 & datasetwithport$final_grade<=10] <- 2
datasetwithport$final_grade[datasetwithport$final_grade>=11 & datasetwithport$final_grade<=15] <- 3
datasetwithport$final_grade[datasetwithport$final_grade>=16 & datasetwithport$final_grade<=20] <- 4

cols <- c("school", "sex", "parent_cohabitation_status", "higher_education", "internet", "final_grade")
datasetwithport[cols] <- lapply(datasetwithport[cols], factor) 
sapply(datasetwithport, class)

## Creating a data partition for the models
set.seed(1)
trn_index = createDataPartition(y = datasetwithport$final_grade, p = 0.80, list = FALSE)
trn_student = datasetwithport[trn_index, ]
tst_student = datasetwithport[-trn_index, ]

##------- Random Forest
set.seed(1)

## Building Random Forest Model - this model excludes the less important variables (higher education + paret_cohabitation_status + internet + sex + school + traveltime + studytime)
student_random_forest_model<-randomForest(final_grade ~. - (first_period_grade + second_period_grade + higher_education + parent_cohabitation_status + internet + sex + school + traveltime + studytime),data = trn_student , ntree = 50000)
predictions_random_forest <- predict(student_random_forest_model, tst_student) 

accuracy_random_forest <- sum(predictions_random_forest == tst_student$final_grade) /
  length(tst_student$final_grade)
accuracy_random_forest

plot(student_random_forest_model, main = 'Error of Grade Prediction by Class')
