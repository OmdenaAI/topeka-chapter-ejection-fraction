from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true[:len(y_pred)], y_pred)
    mae = mean_absolute_error(y_true[:len(y_pred)], y_pred)
    r2 = r2_score(y_true[:len(y_pred)], y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    return mse, mae, r2
