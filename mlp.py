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

# Will be used to analyze the accuracy and plot the ROC curve later.
fold_data = []

fold = 1

for train_index, test_index in skf.split(X, y):
    print(f"\n Fold {fold}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    mlp = MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42+fold,
    early_stopping=True
    )
    mlp.fit(X_train, y_train)

    # Evaluate
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc: .4f}")
    print(classification_report(y_test, y_pred))

    # Save the data necessary to plot the ROC curve
    fold_data.append({
        'accuracy': acc,
        'y_test': y_test,
        'y_proba': mlp.predict_proba(X_test), # Get the probabilites
        'y_train': y_train
    })

    fold += 1

mean_acc = np.mean([f['accuracy'] for f in fold_data])
print(f"Average accuracy after {k} folds: {mean_acc: .4f}")

# Gets the fold with the closest accuracy to the mean
mean_ant = max(
    [f for f in fold_data if f['accuracy'] <= mean_acc],
    key=lambda x: x['accuracy']
)

print(f"Closest accuracy to the average: {mean_ant['accuracy']: .4f}")


# ROC Curve using OvR macro-average
# Calcula a curva ROC pra cada classe e depois tira a média)

# Necessary to plot
label_binarizer = LabelBinarizer().fit(mean_ant['y_train'])
y_onehot_test = label_binarizer.transform(mean_ant['y_test'])

macro_roc_auc_ovr = roc_auc_score(
    mean_ant['y_test'],
    mean_ant['y_proba'],
    multi_class="ovr",
    average="macro",
)

RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        mean_ant['y_proba'].ravel(),
        name=f"Macro-average ROC",
        plot_chance_level=True,
        curve_kwargs={"color": "red"},
    )

plt.savefig("mlp_macro_roc.png")
plt.close()

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.4f}")
