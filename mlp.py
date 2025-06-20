import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("cleaned.csv")

X = df.drop("Air Quality", axis=1)
y = df["Air Quality"]

k = 10  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []

# Will be used to plot the ROC curve of the test with better accuracy.
best_acc = 0
best_y_test = None
best_y_proba = None
best_y_train = None

fold = 1

for train_index, test_index in skf.split(X, y):
    print(f"\n Fold {fold}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scaler // aproxima pra uma distribuicao normal (eu acho)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True
    )
    mlp.fit(X_train, y_train)

    # Evaluate
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc: .4f}")
    #print(classification_report(y_test, y_pred))
    accuracies.append(acc)

    # Save the test with the best score:
    if acc > best_acc:
        best_acc = acc
        best_y_test = y_test
        best_y_proba = mlp.predict_proba(X_test) # Get the probabilites to plot the ROC curve.
        best_y_train = y_train


    fold += 1

print(f"Average accuracy after {k} folds: {np.mean(accuracies): .4f}")
print(f"Best accuracy between folds: {best_acc:.4f}")


# ROC Curve using OvR macro-average
# Calcula a curva ROC pra cada classe e depois tira a média)

# Necessary to plot
label_binarizer = LabelBinarizer().fit(best_y_train)
y_onehot_test = label_binarizer.transform(best_y_test)

macro_roc_auc_ovr = roc_auc_score(
    best_y_test,
    best_y_proba,
    multi_class="ovr",
    average="macro",
)

RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        best_y_proba.ravel(),
        name=f"Macro-average ROC",
        plot_chance_level=True,
        curve_kwargs={"color": "red"},
    )

plt.savefig("mlp_macro_roc.png")
plt.close()

print(f"Macro-averaged One-vs-Rest ROC AUC score of the best fold:\n{macro_roc_auc_ovr:.4f}")
