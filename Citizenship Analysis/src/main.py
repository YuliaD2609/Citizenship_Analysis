from src import models, evaluation, preprocessing
import pandas as pd
import os

def main():
    file_path = '../Dataset/citizenship-datasets.xlsx'  # Specifica il percorso corretto del file
    X, y = preprocessing.load_and_clean_data(file_path)

    save_path = os.path.join('../Output')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    X.to_csv(os.path.join(save_path, f'X_cleaned.csv'), index=False)
    y.to_csv(os.path.join(save_path, f'y_cleaned.csv'), index=False)
    print("Dati preprocessati salvati")

    y = pd.read_csv('../Output/y_cleaned.csv').values.ravel()  # Converti y in un array 1D

    trained_models, X_test, y_test = models.train_models(X, y)
    print("Modelli addestrati")
    pd.DataFrame(X_test).to_csv(os.path.join(save_path, f'X_test.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(save_path, f'y_test.csv'), index=False)
    print("Dati di test salvati")

    y_test = pd.read_csv('../Output/y_test.csv').values.ravel()  # Converti y_test in un array 1D

    results = evaluation.evaluate_models(X_test, y_test)
    results_df = pd.DataFrame(results).T
    print("Dataframe risultante:")
    print(results_df.to_string())


if __name__ == "__main__":
    main()
