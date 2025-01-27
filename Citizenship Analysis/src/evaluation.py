from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(X_test, y_test):
    # Carica i modelli addestrati
    models = {
        'Logistic Regression': joblib.load('../Models/Logistic Regression.pkl'),
        'Random Forest': joblib.load('../Models/Random Forest.pkl'),
        'Hist Gradient Boosting': joblib.load('../Models/Hist Gradient Boosting.pkl'),
        'Decision Tree': joblib.load('../Models/Decision Tree.pkl')
    }

    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    plot_results(results, 'citizenship-results')
    return results


def plot_results(results, name_data: str):
    results_df = pd.DataFrame(results).T

    # Creazione di un grafico a barre per le metriche
    results_melted = results_df.reset_index().melt(id_vars="index",
                                                   var_name="Metric",
                                                   value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_melted, x="Metric", y="Score", hue="index")
    plt.title("Comparison of Model Performance")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()

    # Creazione della directory se non esiste
    save_path = os.path.join('../Output/Plots', name_data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, 'model_performances.png')

    # Salvataggio del grafico
    plt.savefig(file_path)
    print(f"Plot salvato in {file_path}")
    plt.show()

    # Creazione di un istogramma separato per ogni metrica
    save_path_scores = os.path.join(save_path, 'Scores')
    if not os.path.exists(save_path_scores):
        os.makedirs(save_path_scores)

    for metric in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x=results_df.index, y=metric)
        plt.title(f'Comparison of Model Performance - {metric}')
        plt.ylabel(f'{metric} Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Salvataggio dell'istogramma
        metric_file_path = os.path.join(save_path_scores, f'{metric}_performance.png')
        plt.savefig(metric_file_path)
        print(f"Istogramma {metric} salvato in {metric_file_path}")
        plt.show()

    # Creazione di un istogramma separato per ogni modello che rappresenta tutte le metriche
    save_path_models = os.path.join(save_path, 'Models')
    if not os.path.exists(save_path_models):
        os.makedirs(save_path_models)

    for model_name in results_df.index:
        model_scores = results_df.loc[model_name]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_scores.index, y=model_scores.values)
        plt.title(f'{model_name} Performance Metrics')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Salvataggio dell'istogramma per il modello
        model_file_path = os.path.join(save_path_models, f'{model_name}_metrics.png')
        plt.savefig(model_file_path)
        print(f"Istogramma delle metriche per {model_name} salvato in {model_file_path}")
        plt.show()

