###############################################################################################################
#   Name: Mahesh Patil
#   Topic: Telecom Customer Churn Case Study
###############################################################################################################

# Clear the global environment
rm(list = ls())

# Load the required libraries
library(mice)           # For checking missing data and imputation if required
library(dplyr)          # For data manipulation
library(ggcorrplot)     # For Visualizaing correlation plot
library(ggplot2)        # For visualization
library(caret)          # For findCorrelation function AND data partitioning.
library(pROC)           # For model evaluation
library(neuralnet)      # For neural network model
library(e1071)          # For SVM

# Set Working directory
setwd('/Telecom')
data.df<-read.csv('data.csv', header = TRUE, stringsAsFactors = FALSE)

##################################### Data Exploration ##########################################

# Overview of loaded data
str(data.df)
summary(data.df)

# Observing output variable
churn_table<-table(data.df$Churn) 
churn_table
cbind(churn_table, prop.table(churn_table))
# Output variable Churn is highly skewed, as can be seen from table.
# Almost 86% of the customers are non-churners while 14% are churners.
# The predictive model needs to have accuracy more than 86%, to be meaningful. As by just saying every customer is 
# non-churner, we are correct, 86% of the time. Model need to do better than this to be useful.


# It can be seen that State, Area Code, International Plan, Voice Mail Plan, Churn are nominal
# categorical variables.
# So convert these variables from character to factors.

data.df$State<-as.factor(data.df$State)
data.df$Area.code<-as.factor(data.df$Area.code)
data.df$International.plan<-as.factor(data.df$International.plan)
data.df$Voice.mail.plan<-as.factor(data.df$Voice.mail.plan)
data.df$Churn<-as.factor(data.df$Churn)

# Overview the the data again after this change.
str(data.df)

# Check for missing values in dataset
sapply(data.df, function(x) sum(is.na(x)))
md.pattern(data.df)
# Note that Id is a character variable (and its just the Customer ID). So, md.pattern is introducing
# NAs by coercion. We can safely ignore this.
# There are no missing values in this dataset.

#################################### Correlation between numerical variables

# Carve-out numerical and integer variables from the dataset so they can be handled separately.

data.num<-subset(data.df, select = c('Account.length','Number.vmail.messages','Total.day.minutes',
                                     'Total.day.calls', 'Total.day.charge', 'Total.eve.minutes', 'Total.eve.calls',
                                     'Total.eve.charge','Total.night.minutes', 'Total.night.calls', 'Total.night.charge',
                                     'Total.intl.minutes','Total.intl.calls','Total.intl.charge','Customer.service.calls'))

cor1<-cor(data.num)
cor2 <- findCorrelation(cor1, cutoff = 0.001, verbose = FALSE)
cor3 <- cor1[cor2,cor2]
ggcorrplot(cor3, lab = TRUE, type = 'lower')

# From the correlation plot it is clear that minutes variables are highly correlated with the
# Charge variables. In fact the correlation is 1. But among other variables there is no statistically 
# significant relationship.
# It means all the information that is carried by (day, night, eve and intl) charge variables is 
# is also contained in respective minutes variables. So either of them can be used to explain variance 
# in churn. Both are not requred. 

sd(data.df$Total.day.minutes) # standard deviation = 54.46739
sd(data.df$Total.day.charge)  # standard deviation = 9.259435

# The standard deviation of minutes variables is much high compared to charge variables.
# So differences in values will be more visible and appreciable visually if we select 
# minutes varibles and discard charge variables. 
# (However, choosing any of the variables (minutes or charge) is okay.)
# Also, the 'Id' variable is just customer Identification and has no relevance in deciding 
# churn. So removing that variable as well.

# Removal of Correlated variables and 'Id' variable
chargeVar<-c('Id','Total.day.charge','Total.eve.charge','Total.night.charge','Total.intl.charge')
data.df1<-data.df[ , !names(data.df) %in% chargeVar]

################################################## Factors affecting churn
################## (Continuous Data) Numeric and Integer data ############

################ Customer Service calls
# Testing whether number of calls made to customer service has any relationship with churning.
ggplot(data=data.df1, aes(Churn, Customer.service.calls, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Customer Service calls') + ggtitle('Customer Service calls Vs Churn')

# From the boxplot it appears that customers who make more number of calls are likely to churn.
# Verify this with Hypothesis test (t-test for comparing mean of number of customer care calls of churn and non-churn 
# customers)

# t-test can be used to check whether the two sample distributions are coming from the same population or not.
# Therefore, t-test can be used to check whether a particular variable has impact on churning.
# Comparing two distribution of a variable segregated based on (Churn=True and Churn=False) can tell, if these is 
# statistically significant difference between these two distribution. If the difference is significant, it means 
# that variable is important from the point of view of churning.

t.test(data.df1$Customer.service.calls[data.df1$Churn=='FALSE'],data.df1$Customer.service.calls[data.df1$Churn=='TRUE'])
# The p-value of t-test is very small (p-value < 2.2e-16). Hence we reject the null hypothesis that variables are independant.
# It shows there is statistically significant relationship between number of calls made to customer care and churning.
# In fact, on average churning customer makes 2.23 calls as compared to average 1.45 calls made by non-churning customer.

#################### Total day calls

ggplot(data=data.df1, aes(Churn, Total.day.calls, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Day Calls') + ggtitle('Total Day Calls Vs Churn')

ggplot(data=data.df1, aes(x=Total.day.calls, fill=Churn)) + geom_density() + labs(title="Total Day calls Density plot")+
  xlab('Total Day Calls')
ggplot(data=data.df1, aes(x=Total.day.calls, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Day calls Density Plot")+xlab('Total Day Calls')
# Density plot suggests that distribution is nearly normal hence data trnasformation may not be required.
# Linear algorithms, such as regression models, require that data is normally distributed. Normality of data is the
# underlying assumption in these algorithm. Hence, data check for narmality is reuired.
# If data is nor normally distributed then we can transform data using log-transformation or square root transformation
# or reciprocal transformation, etc. In this case, it is not required.
# From the box plot and density plots  it appears that total day calls does not affect churning.

# Verify inference from visual plots with Statistical Hypothesis Test (t-test)

t.test(data.df1$Total.day.calls[data.df1$Churn=='FALSE'],data.df1$Total.day.calls[data.df1$Churn=='TRUE'])
# p-value = 0.3165, P-vale is not small (<0.05). Hence, we fail to reject the null hypothesis that the total day calls 
# and Churning are independent.

##################### Total Day Minutes

ggplot(data=data.df1, aes(Churn, Total.day.minutes, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Day Minutes') + ggtitle('Total Day Minutes Vs Churn')
ggplot(data=data.df1, aes(x=Total.day.minutes, fill=Churn)) + geom_density() + labs(title="Total Day Minutes Density plot")+
  xlab('Total Day Minutes')
ggplot(data=data.df1, aes(x=Total.day.minutes, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Day Minutes Density Plot") + xlab('Total Day Minutes')
# Visual plots suggest that, higher day minutes lead to non-churning. Checking this by conducting t-test again.
# The density plots suggest that, for churning customers, the total day minutes have overlap of two normal distribution.
# Meaning, the total day minutes data for churning customer contains two different distributions.

t.test(data.df1$Total.day.minutes[data.df1$Churn=='FALSE'],data.df1$Total.day.minutes[data.df1$Churn=='TRUE'])
# p-value < 2.2e-16 Indded!!!
# t-test as well as plots suggest that, total day minutes have significant impact on churning.

################### Total Evening Minutes

ggplot(data=data.df1, aes(Churn, Total.eve.minutes, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Evening Minutes') + ggtitle('Total Evening Minutes Vs Churn')
ggplot(data=data.df1, aes(x=Total.eve.minutes, fill=Churn)) + geom_density() + labs(title="Total Evening Minutes Density plot")+
  xlab('Total Evening Minutes')
ggplot(data=data.df1, aes(x=Total.eve.minutes, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Evening Minutes Density Plot") + xlab('Total Evening Minutes')
# From the plots it appears, Total evening minutes affect churning. But this relationship does not appear to be strong.
# Checking this with t-test

t.test(data.df1$Total.eve.minutes[data.df1$Churn=='FALSE'],data.df1$Total.eve.minutes[data.df1$Churn=='TRUE'])
# Hypothesis test suggests that, total evening minutes have statistically significant impact on churning.
# So it is an impotant variable in analysis.

################### Total Evening Calls

ggplot(data=data.df1, aes(Churn, Total.eve.calls, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Evening Calls') + ggtitle('Total Evening Calls Vs Churn')
ggplot(data=data.df1, aes(x=Total.eve.calls, fill=Churn)) + geom_density() + labs(title="Total Evening Calls Density plot")+
  xlab('Total Evening Calls')
ggplot(data=data.df1, aes(x=Total.eve.calls, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Evening Calls Density Plot") + xlab('Total Evening Calls')
# There does nt appear to be significant relationship between total evening calls and churning.
# Testing this assumption with t-test

t.test(data.df1$Total.eve.calls[data.df1$Churn=='FALSE'],data.df1$Total.eve.calls[data.df1$Churn=='TRUE'])
# Statistical test also suggests that, total evening calls do not affect churning.

################### Total Night Minutes

ggplot(data=data.df1, aes(Churn, Total.night.minutes, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Night Minutes') + ggtitle('Total Night Minutes Vs Churn')
ggplot(data=data.df1, aes(x=Total.night.minutes, fill=Churn)) + geom_density() + labs(title="Total Night Minutes Density plot")+
  xlab('Total Night Minutes')
ggplot(data=data.df1, aes(x=Total.night.minutes, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Night Minutes Density Plot") + xlab('Total Night Minutes')
# Total night minutes does not seem to have an impact on churning.
# Checking this with t-test

t.test(data.df1$Total.night.minutes[data.df1$Churn=='FALSE'],data.df1$Total.night.minutes[data.df1$Churn=='TRUE'])
# Statistical hypothesis test suggest that total night minutes does have an impact on churning. 

################### Total Night Calls

ggplot(data=data.df1, aes(Churn, Total.night.calls, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total Night Calls') + ggtitle('Total Night Calls Vs Churn')
ggplot(data=data.df1, aes(x=Total.night.calls, fill=Churn)) + geom_density() + labs(title="Total Night Calls Density plot")+
  xlab('Total Night Calls')
ggplot(data=data.df1, aes(x=Total.night.calls, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total Night Calls Density Plot") + xlab('Total Night Calls')
# Total Night calls does not seem to have impact on churning. Verify this with hypothesis test 

t.test(data.df1$Total.night.calls[data.df1$Churn=='FALSE'],data.df1$Total.night.calls[data.df1$Churn=='TRUE'])
# Hypothesis test also suggest that, total night calls does not affect churning.

##################### Total International Minutes

ggplot(data=data.df1, aes(Churn, Total.intl.minutes, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total International Minutes') + ggtitle('Total International Minutes Vs Churn')
ggplot(data=data.df1, aes(x=Total.intl.minutes, fill=Churn)) + geom_density() + labs(title="Total International Minutes Density plot")+
  xlab('Total International Minutes')
ggplot(data=data.df1, aes(x=Total.intl.minutes, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total International Minutes Density Plot") + xlab('Total International Minutes')
# From the visual plots there does not appear to be any relationship between total international minutes and 
# churning. But it needs to be verified with statistical inference test.

# T-test
t.test(data.df1$Total.intl.minutes[data.df1$Churn=='FALSE'],data.df1$Total.intl.minutes[data.df1$Churn=='TRUE'])
# p-value = 9.066e-05. Hence, we reject the null hypothesis.
# Though the relationship is not visible from boxplot and density plots, however t-test suggests that there is 
# relationship between total international minutes and churning.

###################### Total International Calls

ggplot(data=data.df1, aes(Churn, Total.intl.calls, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Total International Calls') + ggtitle('Total International Calls Vs Churn')
ggplot(data=data.df1, aes(x=Total.intl.calls, fill=Churn)) + geom_density() + labs(title="Total International Calls Density plot")+
  xlab('Total International Calls')
ggplot(data=data.df1, aes(x=Total.intl.calls, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Total International Calls Density Plot") + xlab('Total International Calls')
# From visual plots it seems, if total international calls are on higher side, customer is not likely to churn.
# Checking whether this difference is significant enough.
# The distribution of Total international calls appears to be right skewed. Normalization of data may be required 
# in case of logistic regression.

t.test(data.df1$Total.intl.calls[data.df1$Churn=='FALSE'],data.df1$Total.intl.calls[data.df1$Churn=='TRUE'])
# p-value = 0.003186
# There is statistically significant relationship between total international calls and churning.

################## Number of Vmail Messages

ggplot(data=data.df1, aes(Churn, Number.vmail.messages, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Number of Vmail Messages') + ggtitle('Number of Vmail Messages Vs Churn')
ggplot(data=data.df1, aes(x=Number.vmail.messages, fill=Churn)) + geom_density() + labs(title="Number of Vmail Messages Density plot")+
  xlab('Number of Vmail Messages')
ggplot(data=data.df1, aes(x=Number.vmail.messages, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Number of Vmail Messages Density Plot") + xlab('Number of Vmail Messages')
# Plots suggests that, Number of vmail messages have visually noticable impact on churning. 
# Verifying this with t-test

# From the density plots, it appears that there are two different distributions in the Voice messages dataset.
# Probably, the distributions are due to voice mail plan, YES/NO condition. Checking.
ggplot(data = data.df1, mapping = aes(x=Voice.mail.plan, y = Number.vmail.messages, fill=Voice.mail.plan)) + 
  geom_boxplot() +  xlab('Voice Mail Plan') + ylab('Number of Vmail Messages')
table(data.df1$Voice.mail.plan,data.df1$Number.vmail.messages) 
# Indeed. For NO voice mail plan, the number of Vmail messages are zero, which equals 2411. 
# Hence, there appear two distributions in the density plots above.

t.test(data.df1$Number.vmail.messages[data.df1$Churn=='FALSE'],data.df1$Number.vmail.messages[data.df1$Churn=='TRUE'])
# p-value = 8.765e-09
# P-value is very small, hence we reject the null hypothesis that the two variables are independent.

################# Account length

ggplot(data=data.df1, aes(Churn, Account.length, fill=Churn)) + geom_boxplot() + 
  xlab('Churn') + ylab('Account Length') + ggtitle('Account Length Vs Churn')
ggplot(data=data.df1, aes(x=Account.length, fill=Churn)) + geom_density() + labs(title="Account Length Density plot")+
  xlab('Account Length')
ggplot(data=data.df1, aes(x=Account.length, fill=Churn)) + geom_density() + facet_grid(Churn~ .)+
  labs(title="Account Length Density Plot") + xlab('Account Length')

t.test(data.df1$Account.length[data.df1$Churn=='FALSE'],data.df1$Account.length[data.df1$Churn=='TRUE'])
# p-value = 0.3365. 
# P-value is high, hence we fail to reject null hypothesis that the variables are independant.

############################## Categorical variables
# There are five categorical variables in our data set.
# These are: State, Area code, International plan, Voice mail plan and churn
# Checking interdependance of these variables 

# Dependance of categorical variables can be checked by Chi-Square test for independance. Chi-square test is based on 
# chi square distribution.

################### Checking whether State and Area code are dependant.
chisq.test(data.df1$State, data.df1$Area.code) # p-value = 0.6081
# P-value is high, it means state and Area code are independent.

################### Checking International plan and area code are dependant
mosaicplot(data.df$International.plan~data.df$Area.code)
table(data.df$International.plan, data.df$Area.code)
# Mosaic plot and table suggest loose relationship between the variables.

# Checking by Chi-square test.
chisq.test(data.df$International.plan,data.df$Area.code) # p-value = 0.01891
# P-value is less than0.05, so we reject the null hypothesis that international plan and area code are independant.

######################## Checking Voice mail plan and area code are dependant
mosaicplot(data.df$Voice.mail.plan~data.df$Area.code)
table(data.df$Voice.mail.plan,data.df$Area.code)
# Mosaic plot and table indicate no relationship. Checking by chi-square test.

chisq.test(data.df$Voice.mail.plan,data.df$Area.code)  # p-value = 0.5046
# P-Value is high, so we fail to reject the null hypothesis. So the parameters are independant.

######################## Relationship between Area Code and Churning
mosaicplot(data.df$Churn~data.df$Area.code)
table(data.df$Churn,data.df$Area.code)
# Table and mosiac plot indicate no relationship. Verify by chi-square test

chisq.test(data.df$Churn,data.df$Area.code)  # p-value = 0.9151
# So, there is no relationship between variables.

######################## Relationship between international plan and churning
mosaicplot(data.df$Churn~data.df$International.plan)
table(data.df$Churn,data.df$International.plan)
cbind(table(data.df$Churn,data.df$International.plan),prop.table(table(data.df$Churn,data.df$International.plan)))
# From the mosaic plot and table there appears to be a relationship.
# Checking with chi-square test

chisq.test(data.df$Churn,data.df$International.plan)  # p-value = 2.2e-16
# P-value is very low. Hence, we reject the null hyppothesis that the variables are independent.
# So, the international plan has significant impact on churning.

######################## Relationship between Voice mail plan and churning
mosaicplot(data.df$Churn~data.df$Voice.mail.plan)
table(data.df$Churn,data.df$Voice.mail.plan)
cbind(table(data.df$Churn,data.df$Voice.mail.plan),prop.table(table(data.df$Churn,data.df$Voice.mail.plan)))
# This suggest there is relationship between variables. Testing via hypothesis test.

chisq.test(data.df$Churn,data.df$Voice.mail.plan)  # p-value = 5.151e-09
# This indicate voice mail plan significantly affects churning.

######################## Relationship between State and churning
table(data.df$Churn,data.df$State)
# There are so many states, hence it is difficult to visualiza via mosiac plot or say anything conclusively from
# table data above.
# So, checking via statistical test
chisq.test(data.df$Churn,data.df$State)  # p-value = 0.002296
# Chi-square test done with some warnings. But, apparently it suggests relationship between variables.

######################### Relationship between international plan and voice plan
# From above analysis, it is clear that international plan and voice plan have significant impact on Churning.
# Now, checking whether there is relationship between international plan and voice plan.
mosaicplot(data.df$International.plan~data.df$Voice.mail.plan)
table(data.df$International.plan,data.df$Voice.mail.plan)
# From above, there does not appear any relationship.
# Cheking by chi-square test.
chisq.test(data.df$International.plan,data.df$Voice.mail.plan)  # p-value = 0.7785
# The variables are indeed independant.

##################### Summary of Exploratory Data Analysis ###################################################
# 1. Minutes variables are highly correlated with charge variables. Charge variables can be removed from analysis. 'Id' variable is also removed from
# analysis.
# 
# 2. There are no missing values in dataset.
# 
# 2. Factors affecting customer churn:
# a. Number of customer service calls. If the number of calls to customer care are 3 or more, then customer is highly likely to switch. More number of customer
# care calls indicate that customer is agitated or has some trouble with telecom service.
# b. Total day minutes are important in deciding churning by the customer. Higher the total day minutes (>225 mins), higher of chances of customer switching.
# c. Total evening minutes, Total night minutes are also important factor affecting churning.
# d. Total international calls and total international minutes are important factors affecting churning. Higher the total international minutes and lesser 
# the number of international calls then more likely customer will switch. This clearly indicates international plan is not at all attractive to customers.
# It is one of the main reasons customers are switching.
# e. Customers with voice mail plan are not likely to switch. So, it can be a deciding factor whether custoer will swtich or not.
# f. State of customer is also important factor in churning.
# 
# 3. Factors Not affecting customer churn:
# Total Day Calls, Total evening calls, Total night calls, Account Length, area code.


######################################## Predictive Model #######################################################

# Splitting the data into train, cross-validation and test sets.
set.seed(007)   # For reproducible results.
sampl<-createDataPartition(data.df1$Churn, p = 0.7, list = FALSE)
train<-data.df1[sampl,]
sampl1<-createDataPartition(data.df1[-sampl,]$Churn, p=0.5, list=FALSE)
cv<-data.df1[-sampl,][sampl1,]
test<-data.df1[-sampl,][-sampl1,]

######## Logit Regression
modelCtrl <- trainControl(method = "cv", number = 3)

logit_reg<- train(Churn~., method = "glm", family = "binomial", data = train, trControl = modelCtrl)
summary(logit_reg)
varImp(logit_reg)

logist_pred <- predict(logit_reg, newdata = cv)
logit_acc <- round(1-mean(logist_pred!=cv$Churn),2)
table(actual=cv$Churn, predicted=logist_pred)

logit_roc <- roc(response = cv$Churn, predictor = as.numeric(logist_pred))
plot(logit_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)

####### Decision Tree
dt.model <- train(Churn ~ ., method = "rpart2", data = train, trControl = modelCtrl)
dt.pred <- predict(dt.model, newdata = cv)
cart.accuracy <- 1-mean(dt.pred != cv$Churn)
table(actual=cv$Churn, predicted=dt.pred)

dt_roc <- roc(response = cv$Churn, predictor = as.numeric(dt.pred))
plot(dt_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)

###### Bagging
bagging <- train(Churn ~ ., method = "treebag", data = train, trControl = modelCtrl)
bagging_pred <- predict(bagging, newdata = cv)
bagging_roc <- roc(response = cv$Churn, predictor = as.numeric(bagging_pred))
plot(bagging_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)

###### Boosting
boosting<-train(Churn ~ ., method = "C5.0", data = train)
boosting_pred <- predict(boosting, newdata = cv)
boosting_roc <- roc(response = cv$Churn, predictor = as.numeric(boosting_pred))
plot(boosting_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)

###### Random Forest
rf<-train(Churn ~ ., method = "rf", data = train, trControl = modelCtrl)
rf_pred <- predict(rf, newdata = cv)
rf_roc <- roc(response = cv$Churn, predictor = as.numeric(rf_pred))
plot(rf_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)




rf.pred2 <- predict(rf, newdata = cv, type = "prob")

rf2.pred <- factor(ifelse( rf.pred2[,2] > 0.3, "TRUE", "FALSE"), levels = levels(cv$Churn))
rf2.accuracy <- round(1-mean(rf2.pred != cv$Churn),2)
confusionMatrix(rf2.pred, cv$Churn)



rf_roc2 <- roc(response = cv$Churn, predictor = as.numeric(rf2.pred))
plot(rf_roc2,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)


###### Bench
bench<-ifelse(rf_pred==TRUE,FALSE,FALSE)
bench_roc <- roc(response = cv$Churn, predictor = as.numeric(bench))
plot(bench_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)

###### SVM
temp<-as.formula(paste("Churn ~", paste(names(train)[!names(train) %in% "Churn"], collapse = " + ")))


model_svm <- svm(Churn~., train)
pred <- predict(model_svm, cv)
svm_roc <- roc(response = cv$Churn, predictor = as.numeric(pred))
plot(svm_roc,legacy.axes = TRUE, print.auc.y = 0.4, print.auc = TRUE)


