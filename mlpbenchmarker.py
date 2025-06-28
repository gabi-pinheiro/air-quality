import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
# benchmarking
import os
import time
import argparse
import numpy as np
from sklearn import metrics


def saveRocCurve(y_onehot_test, y_score, macro_roc_auc_ovr):
    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name=f"Macro-average ROC",
        color= "red",
        plot_chance_level=True,
    )
    plt.savefig("MLP_macro_roc.png")
    plt.close()

def saveResults(results, args):
    # Calculate averages
    averages = results.mean(numeric_only=True).to_frame().T 
    averages['Run'] = 'Average'

    # Add to the final file, save as .csv
    final_results = pd.concat([averages, results], ignore_index=True)
    final_results.to_csv(args.metric, index=False)
    
    # Summary just because
    print("\nBenchmark Summary:")
    print(f"Average training time: {averages['training_time'].values[0]:.4f} seconds")
    print(f"Average testing time: {averages['testing_time'].values[0]:.4f} seconds")
    print(f"Average total time: {averages['total_time'].values[0]:.4f} seconds")
    print(f"Average accuracy: {averages['accuracy'].values[0]:.4f}")
    print(f"Average F-measure: {averages['f-measure'].values[0]:.4f}")

def runBenchmark(X, y, args):
    MAX_RUN = args.n__number_runs
    R_FOLDS = 10

    print(f"Executará {MAX_RUN} iterações.")

    # copiado e adaptado do de AD
    results = []
    roc_values = {
        'run': [],
        'roc_auc_ovr': [],
        'y_score': [],
        'y_onehot': [],
    }

    # Split into 70/30 (or whichever desired) before proceeding
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=39)


    # Do runs
    print(f"Executing {MAX_RUN} runs:")
    for i in range(MAX_RUN):
        # Store the fold metrics for each iteration
        fold_metrics = {
            'training_time': [],
            'testing_time': [],
            'accuracy': [],
            'f_measure': [],
            'roc_auc_ovr': [],
            'y_score': [],
            'y_onehot': [],
        }
        
        # print(f"Creating {R_FOLDS}-fold cross-validation.")
        skf = StratifiedKFold(n_splits=R_FOLDS)
        # print(f"Starting run {i+1}:")
        for train_index, test_index in skf.split(X_train, y_train): 
            fold_X_train, fold_X_test = X_train[train_index], X_train[test_index]
            fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]
            
            mlp = MLPClassifier(
                hidden_layer_sizes=(64,32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=39,
                early_stopping=True
            )

            # Benchmark training
            start_training = time.time()
            mlp.fit(fold_X_train, fold_y_train)
            training_time = time.time() - start_training

            # Benchmark testing
            start_testing = time.time()
            y_pred = mlp.predict(fold_X_test)
            testing_time = time.time() - start_testing

            # Metrics
            acc = metrics.accuracy_score(fold_y_test, y_pred)
            f_measure = metrics.f1_score(fold_y_test, y_pred, average='weighted')

            # ROC Curve
                # Initialize
            y_score = mlp.predict_proba(fold_X_test)
            label_binarizer = LabelBinarizer().fit(fold_y_train)
            y_onehot_test = label_binarizer.transform(fold_y_test)
                # Using OvR macro-average
            macro_roc_auc_ovr = roc_auc_score(
                fold_y_test,
                y_score,
                multi_class="ovr",
                average="macro",
            )
            # These fold metrics are for deciding the average ROC curve from the folds.
            # print(f"\t\tRan a fold: {train_index+1}")
            fold_metrics['roc_auc_ovr'].append(macro_roc_auc_ovr)
            fold_metrics['y_score'].append(y_score)
            fold_metrics['y_onehot'].append(y_onehot_test)


            # Store the fold metrics into the array
            fold_metrics['training_time'].append(training_time)
            fold_metrics['testing_time'].append(testing_time)
            fold_metrics['accuracy'].append(acc)
            fold_metrics['f_measure'].append(f_measure)

        # Select the average ROC curve for the fold based on the fold's accuracy
        mean_accuracy = np.mean(fold_metrics['accuracy'])
        # print(f"\tIteration {i+1}'s avg accuracy was: {mean_accuracy}")
            # list all accuracies with acc <= mean_acc
        lower_accuracies = [
            acc for acc in fold_metrics['accuracy'] if acc <= mean_accuracy
        ]
        closest_lower = max(lower_accuracies)
            # for entries with same acc (possible)
        closest_lower_items = [
            i for i, acc in enumerate(fold_metrics['accuracy']) if acc == closest_lower
        ]
        closest_lower_index = closest_lower_items[0]
        # print(f"\t\tThis iteration's closest acc was: {fold_metrics['accuracy'][closest_lower_index]}")
            # Save to this iteration's avg ROC
        roc_values['run'].append(i+1)
        roc_values['roc_auc_ovr'].append(fold_metrics['roc_auc_ovr'][closest_lower_index])
        roc_values['y_score'].append(fold_metrics['y_score'][closest_lower_index])
        roc_values['y_onehot'].append(fold_metrics['y_onehot'][closest_lower_index])

        # Save results
        results.append({
            'run': i+1,
            'training_time': np.mean(fold_metrics['training_time']),
            'testing_time': np.mean(fold_metrics['testing_time']),
            'total_time': np.mean(fold_metrics['training_time']) + np.mean(fold_metrics['testing_time']),
            'accuracy': mean_accuracy,
            'f-measure': np.mean(fold_metrics['f_measure']),
        })

    # Generate our average ROC
    # Select the average ROC curve for the iteration based on their roc_auc_ovr score
    mean_accuracy = np.mean(roc_values['roc_auc_ovr'])
        # list all accuracies with acc <= mean_acc
    lower_accuracies = [
        acc for acc in roc_values['roc_auc_ovr'] if acc <= mean_accuracy
    ]
    closest_lower = max(lower_accuracies)
        # for entries with same acc (possible)
    closest_lower_items = [
        i for i, acc in enumerate(roc_values['roc_auc_ovr']) if acc == closest_lower
    ]
    closest_lower_index = closest_lower_items[0]
    # Create the roc image
    print(f"Creating ROC curve with the values from the Average Iteration's Average Fold results, from run {roc_values['run'][closest_lower_index]}")
    saveRocCurve(roc_values['y_onehot'][closest_lower_index], roc_values['y_score'][closest_lower_index], roc_values['roc_auc_ovr'][closest_lower_index])


    # Send to final processing
    saveResults(pd.DataFrame(results), args)




def main():
    parser = argparse.ArgumentParser(
        description="MLP",
        )
    parser.add_argument('-t', '--test',
                        type=float,
                        help='Test subset ratio size. Default is 0.3 for 30%% test, 70%% train.',
                        default=0.3
                        )
    parser.add_argument('-n' '--number-runs',
                        type=int,
                        help='How many runs to do when executing "--benchmark". Default is 250.',
                        default=250
                        )
    parser.add_argument('-m', '--metric',
                        help='Path to the metrics output file. Default is \'MLP_metrics.csv\'.',
                        default='MLP-metrics.csv'
                        )
    args = parser.parse_args()


    df = pd.read_csv("cleaned.csv")
    attributes =['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']


    X = df[attributes].values
    y = df['Air Quality'].values


    runBenchmark(X, y, args)


    #print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")


if __name__ == '__main__':
    main()