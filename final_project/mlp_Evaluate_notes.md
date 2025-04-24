```python
def create_and_evaluate_model(hidden_units, X_train, y_train, X_valid, y_valid, X_test, y_test):
    tf.keras.backend.clear_session()

    # Input layer
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x = inputs

    # Hidden layers
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1)(x)

    # Create and compile model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mae"])

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_valid, y_valid),
        verbose=0
    )

    # Evaluate - use try-except to handle potential NaN values
    try:
        _, mae = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test).flatten()

        # Check for NaN values and handle them
        if isnan(y_pred).any():
            print(f"Warning: NaN predictions found in architecture {hidden_units}")
            # Remove NaN values for metrics calculation
            mask = ~isnan(y_pred)
            if mask.sum() > 0:  # If we have any non-NaN values
                mse = mean_squared_error(y_test[mask], y_pred[mask])
                r2 = r2_score(y_test[mask], y_pred[mask])
            else:
                mse = nan
                r2 = nan
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Error evaluating model with architecture {hidden_units}: {e}")
        mae = nan
        mse = nan
        r2 = nan

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "history": history.history
    }

# Test multiple architectures
architectures = [
    [16],
    [32],
    [64],
    [128],
    [256],
    [32, 16],
    [64, 32],
    [128, 64],
    [64, 32, 16],
]

results = []
for arch in architectures:
    print(f"Testing architecture: {arch}")
    result = create_and_evaluate_model(arch, X_train, y_train, X_valid, y_valid, X_test, y_test)
    results.append({
        "architecture": str(arch),
        "mse": result["mse"],
        "mae": result["mae"],
        "r2": result["r2"]
    })

# Display results as a table
results_df = DataFrame(results)
print("\nResults comparison:")
print(results_df)

# Plot comparison - only include rows without NaN values
valid_results = results_df.dropna()
if len(valid_results) > 0:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(valid_results["architecture"], valid_results["r2"])
    plt.title("R² Score by Architecture")
    plt.xlabel("Architecture")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(valid_results["architecture"], valid_results["mse"])
    plt.title("MSE by Architecture")
    plt.xlabel("Architecture")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
else:
    print("No valid results to plot comparisons")

# Cross-validation comparison with Linear Regression
print("\nPerforming cross-validation comparison...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mlp_cv_scores = []
lr_cv_scores = []

for train_idx, val_idx in kf.split(X_scaled):
    # Get fold data
    X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train and evaluate MLP with best architecture
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(X_fold_train.shape[1],))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    fold_mlp = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    fold_mlp.compile(loss="mean_squared_error", optimizer="sgd")
    fold_mlp.fit(X_fold_train, y_fold_train, epochs=50, verbose=0)

    try:
        y_pred_fold_mlp = fold_mlp.predict(X_fold_val).flatten()
        # Check for NaN values
        if isnan(y_pred_fold_mlp).any():
            print("Warning: NaN values in MLP predictions for this fold")
            # Use only non-NaN predictions
            mask = ~isnan(y_pred_fold_mlp)
            if mask.sum() > 0:
                mlp_cv_scores.append(r2_score(y_fold_val.iloc[mask], y_pred_fold_mlp[mask]))
            else:
                mlp_cv_scores.append(nan)
        else:
            mlp_cv_scores.append(r2_score(y_fold_val, y_pred_fold_mlp))
    except Exception as e:
        print(f"Error in MLP cross-validation: {e}")
        mlp_cv_scores.append(nan)

    # Train and evaluate Linear Regression
    lr = LinearRegression()
    lr.fit(X_fold_train, y_fold_train)
    y_pred_fold_lr = lr.predict(X_fold_val)
    lr_cv_scores.append(r2_score(y_fold_val, y_pred_fold_lr))

# Filter out NaN scores
mlp_cv_scores_filtered = [score for score in mlp_cv_scores if not isnan(score)]

if mlp_cv_scores_filtered:
    print(f"MLP Cross-Validation R² scores: {mlp_cv_scores}")
    print(f"MLP Cross-Validation Mean R²: {mean(mlp_cv_scores_filtered):.4f}")
else:
    print(f"MLP Cross-Validation R² scores: {mlp_cv_scores}")
    print("No valid MLP cross-validation scores")

print(f"Linear Regression Cross-Validation R² scores: {lr_cv_scores}")
print(f"Linear Regression Cross-Validation Mean R²: {mean(lr_cv_scores):.4f}")```