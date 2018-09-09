library(xts)
library(zoo)
library(ggplot2)
library(forecast)
library(Hmisc)
library(caret)
library(dplyr)
library(neuralnet)
library(nnet)
library(TTR)
library(DataCombine)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(gbm)

setwd("/Users/anon/NUS_Files/DataAnalytics/Day4/")

bike_data <- read.csv("day.csv")

#Converting to date object
bike_data$dteday <- as.POSIXct(bike_data$dteday)

dates <- bike_data$dteday
bike_data$dteday<- NULL

###creating a time series (xts) object ordering by the second column which have the time values
bike_ts <- zoo(bike_data, as.Date(dates,"%Y-%m-%d"), frequency = 4)

#viewing the time series object
head(bike_ts)

##plotting the number of customers (daily)
#can observe an increasing trend from first year to second year
#there is a strong seasonal pattern - with demand peaking in June-August
autoplot(bike_ts[,"cnt"]) + ggtitle("Daily customers for Bikeshare") + xlab("Date") + ylab("No of customers")


#Relation between temperature and Count of customers
plot(bike_ts[,c("temp","hum","windspeed","cnt")], xlab="Date", col='blue',ylab=c("Temp in Celsius","Humidity","Windspeed","Count"), main="Comparison of temperature and number of customers")

#Almost linear relation ship between temperature and count of customers
ggplot(as.data.frame(bike_ts), aes(x= temp,y=cnt))+geom_point(colour = "blue")+labs(x="Temperature (degree celsius)", y="Count of customers", title ="Temperature vs Customer count")

#Not a strong relationship between humidity/ windspeed and count
ggplot(as.data.frame(bike_ts), aes(x= windspeed,y=cnt))+geom_point(colour = "blue")+labs(x="Windspeed", y="Count of customers", title ="Windspeed vs Customer count")

ggplot(as.data.frame(bike_ts), aes(x= hum,y=cnt))+geom_point(colour = "blue")+labs(x="Humidity", y="Count of customers", title ="Humidity vs Customer count")

###observing the relationship between casual and registered users over week(end)days
##Casual users are definitely higher on weekends while registered users are marginally higher on weekdays
ggplot(as.data.frame(bike_ts), aes(x= weekday,y=registered))+geom_point(colour = "blue")+labs(x="weekday", y="Registered customers", title ="Humidity vs Customer count")

ggplot(as.data.frame(bike_ts), aes(x= weekday,y=casual))+geom_point(colour = "blue")+labs(x="weekday", y="Casual customers", title ="Humidity vs Customer count")

###A note about the weekday codes
##weekday 6 is a saturday and 0 is a sunday


###### Data preparation for Modelling#######

###creating a data frame
bikedata_df <- read.csv("day.csv")

#Converting to date object
bikedata_df$dteday <- as.POSIXct(bikedata_df$dteday)

###dealing with outliers
###hurrican sandy on oct 29, 30th 2012
##april 22nd 2012 snow unusual

###finding the difference of successive rows
#bikedata_df$difference[2:nrow(bikedata_df)] <- tail(bikedata_df$cnt,-1)-head(bikedata_df$cnt,-1)

##finding the standard deviation of the differences
## there were two instances where the standard deviation was higher than 4000. and these corresponded
##to the above mentioned natural events

#bikedata_df$sd <- rollapply(bikedata_df[,"difference"], width=3, FUN = sd, na.pad=TRUE)

###removing the rows corresponding to these events
bikedata_df <- bikedata_df[!(bikedata_df$dteday=="2012-04-22"|bikedata_df$dteday=="2012-10-29"|bikedata_df$dteday=="2012-10-30"),]

###Extracting the count, casual and registered columns
demand <- bikedata_df[,c("casual","registered","cnt")]

#creating a lag variable for cnt and registered
demand$lag_cnt <- Lag(demand$cnt, -2)
demand$lag_regd <- Lag(demand$registered, -2)


## for the time being a simpler past 7 days average irrespective of weekday/end. But 
##separate for casual and registered


bikedata_df[,"regdaverage"] <- SMA(bikedata_df[,"registered"],7)

###creating trend variables for temperature - past 7 day average

bikedata_df[, "temp_average"] <- SMA(bikedata_df[,"temp"],7)
bikedata_df[, "atemp_average"] <- SMA(bikedata_df[,"atemp"],7)

##Creating trend variables for other parameters - instead of today's, giving tomorrow's parameter
##as input. The intuition is that tomorrows weather will be a better predictor of day after tomorrows
##weather than todays weather

bikedata_df[,"tomrw_atemp"] <- Lag(bikedata_df$atemp,-1)
bikedata_df[,"tomrw_temp"] <- Lag(bikedata_df$temp,-1)

bikedata_df[,"tomrw_weather"] <- Lag(bikedata_df$weathersit,-1)
bikedata_df[,"tomrw_windspeed"] <- Lag(bikedata_df$windspeed,-1)
bikedata_df[,"tomrw_hum"] <- Lag(bikedata_df$hum,-1)


##creating variable for representing the day of week two days hence
##this is because there is marked difference between weekend and weekday demand
##so in the input variable there should be some indication that the day for which prediction is 
##made is a weekday or weekend

bikedata_df$predict_day <- ifelse(bikedata_df$weekday==5|bikedata_df$weekday==4,0,1)

##similarly giving an indication of whether the day to be predicted (2 days hence) 
##is a working day or not
##done by the Lag command 
bikedata_df$predict_workday <- Lag(bikedata_df$workingday, -2)

##similarly for holiday
bikedata_df$predict_holiday <- Lag(bikedata_df$holiday, -2)


###differencing the variable cnt, casual and registered
bikedata_df$regd_diff <-  c(NA, NA, NA, NA, NA, NA, NA, diff(bikedata_df$registered, lag=7))


######Combining the demand df and bikedata df
bikedata_df <- cbind(bikedata_df, demand)
bikedata_df <- na.omit(bikedata_df)

bikedata_df[,c(14,15,16)] <- NULL

#Making the dataframe ready for modelling
bikedata_df_model <- bikedata_df[,!colnames(bikedata_df) %in%  c("instant","dteday","yr","cnt","casual","difference","sd","cnt_perc_change","holiday","workingday","weekday","temp","atemp","windspeed","weathersit","hum")]

bikedata_df_model <- bikedata_df_model[, !colnames(bikedata_df_model) %in%  c("cnt_diff","casual_diff","cnt_trend","lag_cnt","casual_trend","lag_casual","perc_change","casualaverage", "naive_pred","cntaverage","regd_trend")]


#######Custom function for evaluation ########
evaluation = function(prediction, test){
  accuracy_measure <<- data.frame(cbind(test, prediction))
  error <- RMSE(test, prediction, na.rm = TRUE)
}
################

#########Modelling#######

#Split into training and test sets
set.seed(1234)

train_set <- bikedata_df_model[1:539,]
test_set <- bikedata_df_model[540:719,]


##Fitting the linear model
model = lm(lag_regd ~ ., data = train_set)
summary(model)

##Predicting and creating a dataframe with test set count values and predicted values
#pred_cnt <- predict(model, test_set[,!colnames(test_set)%in% c("lag_cnt")])
pred_regd <- data.frame(prediction = predict(model, test_set))
error = evaluation(pred_regd, test_set$lag_regd)

####### with significant predictors - version 1
model_v1 = lm(lag_regd ~ mnth+regdaverage+tomrw_weather+tomrw_hum+predict_day+predict_holiday+predict_workday+registered, data = train_set)
summary(model_v1)
pred_regd_v1 <- data.frame(prediction = predict(model_v1, test_set))
error = evaluation(pred_regd_v1, test_set$lag_regd)

write.csv(pred_regd_v1,"pred_regd.csv")

###Neural network model######################

#Making the dataframe ready for modelling
#bikedata_df_model_nn <- bikedata_df[,!colnames(bikedata_df) %in%  c("instant","dteday","yr","casual","registered","difference","sd","lag_cnt","cnt_perc_change","holiday","workingday","weekday","temp","atemp","windspeed","weathersit","hum")]
bikedata_df_model_nn <- bikedata_df_model

set.seed(4321)
maxs <- apply(bikedata_df_model_nn,2,max)
mins <- apply(bikedata_df_model_nn,2,min)
scaled <- as.data.frame(scale(bikedata_df_model_nn,center = mins, scale = maxs - mins))

train_set_nn <- scaled[1:539,]
test_set_nn <- scaled[540:nrow(bikedata_df_model_nn),]

numFolds <- trainControl(method = 'cv', number = 10, classProbs = FALSE, verboseIter = TRUE, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

regd.fit <- train(lag_regd ~ ., data = train_set_nn, method = 'nnet', trControl = numFolds, tuneGrid=expand.grid(size=c(8,3), decay=c(0.01,0.01)))
predict_testNN = predict(regd.fit,newdata = test_set_nn)

predict_testNN_descaled = (predict_testNN * (max(bikedata_df_model_nn$lag_regd) - min(bikedata_df_model_nn$lag_regd))) + min(bikedata_df_model_nn$lag_regd)

test_result <- (test_set_nn$lag_regd *(max(bikedata_df_model_nn$lag_regd) - min(bikedata_df_model_nn$lag_regd))) + min(bikedata_df_model_nn$lag_regd)

nn_mse <- sqrt(sum((test_result-predict_testNN_descaled)^2)/length(test_result))

write.csv(predict_testNN_descaled, "predictregd_NN.csv")
#####################

############Decision Trees########

fit <- rpart(lag_regd ~ .,
             data=train_set,minsplit=30,xval=2,
             method="anova")
dt_pred <- predict(fit, test_set)
error <- RMSE(dt_pred, test_set$lag_regd, na.rm = TRUE)
error


#######Random forest model#####
set.seed(1000)
bikedata_df_model_rf <- bikedata_df_model

set.seed(1234)

train_set_rf <- bikedata_df_model_rf[1:539,]
test_set_rf <- bikedata_df_model_rf[540:719,]

regdfit_rf <- randomForest(lag_regd ~ ., data=train_set_rf, importance = TRUE, ntree=3000)

varImpPlot(regdfit_rf)

rf_prediction <- predict(regdfit_rf, test_set_rf)
error <- evaluation(rf_prediction, test_set_rf$lag_regd)
error

write.csv(rf_prediction, "predictregd_RF.csv")

#########Gradient boosting ######
set.seed(1234)

train_set_gbm <- bikedata_df_model_rf[1:358,]
test_set_gbm <- bikedata_df_model_rf[540:719,]

regdfit_gbm <- gbm(lag_regd ~ season+regdaverage+tomrw_weather+predict_workday+registered, data=train_set_gbm,n.trees = 2000)

pred_gbm <- predict(regdfit_gbm, test_set_gbm, n.trees = 2000)
error <- RMSE(pred_gbm, test_set_gbm$lag_regd, na.rm = TRUE)
error



############