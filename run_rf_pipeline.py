# run_rf_pipeline.py
from code.config import DATA_PATH, TARGET_COLUMN, CLEANING_PARAMS, RF_PARAM_GRID, MODEL_SAVE_DIR
from code.data_utils import load_data, clean_data_manual
from code.training_rf import tune_evaluate_and_analyze_rf_pipeline

def run_rf_pipeline():
    # 1. 加载数据
    X, y = load_data(DATA_PATH, target_column=TARGET_COLUMN)
    
    # 2. 数据清洗
    X_clean, y_clean = clean_data_manual(X, y, **CLEANING_PARAMS)
    
    # 3. 运行 RF 全流程
    best_model, metrics, feature_imp_df, top_features = tune_evaluate_and_analyze_rf_pipeline(
        X_data=X_clean,
        y_data=y_clean,
        param_grid=RF_PARAM_GRID,
        model_filename='rf_cleaned_data_tuned',
        top_n=50,
        model_dir=MODEL_SAVE_DIR,
        force_run=False,
        show_plot=True
    )
    
    print(f"\n✅ RF 流程完成。最终 R² = {metrics['R2']:.4f}")

if __name__ == "__main__":
    run_rf_pipeline()