import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned.csv")

X = df.drop("Air Quality", axis=1)
y = df["Air Quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ROC Curve using OvR macro-average
# (Calcula a curva ROC pra cada classe e depois tira a média)

y_proba = mlp.predict_proba(X_test) # Get the probabilities

# Necessary to plot
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

macro_roc_auc_ovr = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="macro",
)

RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_proba.ravel(),
        name=f"Macro-average ROC",
        plot_chance_level=True,
        curve_kwargs={"color": "red"},
    )

plt.savefig("mlp_macro_roc.png")
plt.close()

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")
