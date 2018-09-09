prediction_casual <- read.csv("pred_casual.csv")
prediction_regd <- read.csv("pred_regd.csv")
bikedata <- read.csv("Bike_data_processed.csv")

prediction_nn_regd <- read.csv("predictregd_NN.csv")
prediction_nn_casual <- read.csv("predictcasual_NN.csv")

prediction_rf_regd <- read.csv("predictregd_RF.csv")
prediction_rf_casual <- read.csv("predictcasual_RF.csv")

overall_profit <- data.frame(casual_pred = prediction_casual$prediction, regd_pred = prediction_regd$prediction, count = bikedata[359:719, "cnt"] , lag_count = bikedata[359:719, "lag_cnt"])

overall_profit$casual_pred_nn <- prediction_nn_casual$V1
overall_profit$reg_pred_nn <- prediction_nn_regd$V1

overall_profit$casual_pred_rf <- prediction_rf_casual$x
overall_profit$reg_pred_rf <- prediction_rf_regd$x


####### Calculating profit######

overall_profit$total_pred_lm <- overall_profit$casual_pred + overall_profit$regd_pred
overall_profit$total_pred_nn <- overall_profit$casual_pred_nn + overall_profit$reg_pred_nn
overall_profit$total_pred_rf <- overall_profit$casual_pred_rf + overall_profit$reg_pred_rf

overall_profit <- mutate(overall_profit, naive_profit = pmin(count, lag_count)*2.2 - count*2)
overall_profit <- mutate(overall_profit, lm_model_profit = pmin(total_pred_lm, lag_count)*2.2 - total_pred_lm*2)
overall_profit <- mutate(overall_profit, nn_model_profit = pmin(total_pred_nn, lag_count)*2.2 - total_pred_nn*2)
overall_profit <- mutate(overall_profit, rf_model_profit = pmin(total_pred_rf, lag_count)*2.2 - total_pred_rf*2)


#####Performing an ensemble

ensemble_model <- lm(lag_count ~ total_pred_lm+total_pred_nn+total_pred_rf, data=overall_profit)
ensemble_pred <- predict(ensemble_model, overall_profit[,c("total_pred_lm","total_pred_nn","total_pred_rf")])
overall_profit$ensemble_prediction <- ensemble_pred

overall_profit <- mutate(overall_profit, ensemble_model_profit = pmin(ensemble_prediction, lag_count)*2.2 - ensemble_prediction*2)

#######Final profit numbers
print(sum(overall_profit$ensemble_model_profit))
print(sum(overall_profit$naive_profit))
print(sum(overall_profit$ensemble_model_profit)*100/(sum(overall_profit$ensemble_prediction)*2))
print(sum(overall_profit$naive_profit)*100/(sum(overall_profit$count)*2))
