# Load the data
meat_data <- read.csv("data/tecator.csv")

# Set seed for reproducibility
set.seed(123)

# Extract features and target variable
absorbance_features <- meat_data[, 2:101]  
fat_content <- meat_data$Fat       

# Split data into training (50%) and testing (50%)
train_indices <- sample(1:nrow(meat_data), size = nrow(meat_data) * 0.5)
train_features <- absorbance_features[train_indices, ]
train_target <- fat_content[train_indices]
test_features <- absorbance_features[-train_indices, ]
test_target <- fat_content[-train_indices]

# Standardize features
train_features_scaled <- scale(train_features)
test_features_scaled <- scale(test_features)

# Linear Regression
linear_model <- lm(train_target ~ ., data = as.data.frame(cbind(train_target, train_features_scaled)))
train_predictions <- predict(linear_model, newdata = as.data.frame(train_features_scaled))
test_predictions <- predict(linear_model, newdata = as.data.frame(test_features_scaled))
mse_train <- mean((train_target - train_predictions)^2)
mse_test <- mean((test_target - test_predictions)^2)
cat("Training MSE:", mse_train, "\n")
cat("Testing MSE:", mse_test, "\n")

# Load glmnet library
library(glmnet)
train_features_matrix <- as.matrix(train_features_scaled)
train_target_vector <- as.vector(train_target)

# LASSO Regression
lasso_model <- glmnet(train_features_matrix, train_target_vector, alpha = 1)
plot(lasso_model, xvar = "lambda", label = TRUE, col = c("blue", "green", "purple"))
title("LASSO Coefficients vs Log(Lambda)")

# Lambda with 3 non-zero features
lambda_values <- lasso_model$lambda
for (lambda in lambda_values) {
  coef_nonzero <- coef(lasso_model, s = lambda)[-1]
  num_features <- sum(coef_nonzero != 0)
  if (num_features == 3) {
    cat("Lambda for 3 features:", lambda, "\n")
    break
  }
}

# Extract and display coefficients for 3-feature lambda
selected_coefficients <- coef(lasso_model, s = lambda)
selected_coefficients_matrix <- as.matrix(selected_coefficients)
nonzero_indices <- which(selected_coefficients_matrix != 0)
selected_features <- data.frame(
  Feature = rownames(selected_coefficients_matrix)[nonzero_indices],
  Coefficient = selected_coefficients_matrix[nonzero_indices]
)
print(selected_features)

# Ridge Regression
ridge_model <- glmnet(train_features_matrix, train_target_vector, alpha = 0)
plot(ridge_model, xvar = "lambda", label = TRUE, col = c("orange", "cyan", "pink"))
title("Ridge Coefficients vs Log(Lambda)")

# Cross-Validation for LASSO
cv_lasso <- cv.glmnet(train_features_matrix, train_target_vector, alpha = 1)
plot(cv_lasso, col = "darkgreen")
title("Cross-Validation for LASSO: Error vs Log(Lambda)")
optimal_lambda <- cv_lasso$lambda.min
cat("Optimal Lambda:", optimal_lambda, "\n")

# Non-zero features for optimal lambda
optimal_coefficients <- coef(cv_lasso, s = "lambda.min")
nonzero_features_count <- sum(optimal_coefficients != 0) - 1
cat("Number of non-zero features at optimal lambda:", nonzero_features_count, "\n")

# Test performance for optimal lambda
lasso_test_predictions <- predict(cv_lasso, s = optimal_lambda, newx = as.matrix(test_features_scaled))
mse_test_lasso <- mean((test_target - lasso_test_predictions)^2)
cat("Test MSE for LASSO with Optimal Lambda:", mse_test_lasso, "\n")

# Comparison with log(lambda) = -4
lambda_at_log_minus4 <- exp(-4)
test_predictions_log_minus4 <- predict(lasso_model, s = lambda_at_log_minus4, newx = as.matrix(test_features_scaled))
mse_test_log_minus4 <- mean((test_target - test_predictions_log_minus4)^2)
cat("Test MSE for LASSO with log(lambda) = -4:", mse_test_log_minus4, "\n")

# Scatter plot for optimal lambda
plot(test_target, lasso_test_predictions, 
     main = "Scatter Plot: Actual vs Predicted (Optimal Lambda)", 
     xlab = "Actual Fat", ylab = "Predicted Fat", pch = 19, col = "blue")
abline(0, 1, col = "red")
