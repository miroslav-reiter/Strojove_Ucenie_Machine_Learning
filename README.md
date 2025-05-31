# 🤖 Online kurz: Strojové učenie (Machine Learning) v Pythone so scikit-learn

> Praktický kurz pre začiatočníkov – Regresia v ML, scikit-learn, modelovanie, tréning a vyhodnocovanie

---

## 📘 Obsah kurzu

01. [**🔍 Úvod do strojového učenia a regresie**](#uvod-ml-regresia)  
02. [**📈 Lineárna regresia v scikit-learn**](#linearna-regresia)  
03. [**🧮 Viacnásobná regresia a výber premenných**](#viacnasobna-regresia)

---

<a name="uvod-ml-regresia"></a>
## 🔍 1. Úvod do strojového učenia a regresie

- Vysvetlenie základných pojmov: supervised learning, tréning, testovanie
- Rozdiel medzi regresiou a klasifikáciou
- Typy regresií (lineárna, viacnásobná, polynomiálna...)
- Prehľad knižnice scikit-learn a jej moduly
- Príprava datasetu: rozdelenie na tréning/test, čistenie, transformácia

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

# Načítanie dát
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

<a name="linearna-regresia"></a>
## 📈 2. Lineárna regresia v scikit-learn

- Teória: čo je lineárna rovnica, koeficienty a intercept
- Tréning modelu: `LinearRegression()`
- Vyhodnotenie: R², MSE
- Vizualizácia predikcie

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train[['bmi']], y_train)

preds = model.predict(X_test[['bmi']])
print("R2:", r2_score(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
```

---

<a name="viacnasobna-regresia"></a>
## 🧮 3. Viacnásobná regresia a výber premenných

- Vysvetlenie konceptu: viacero vstupov (features)
- Normalizácia: `StandardScaler`
- Výber relevantných premenných: `SelectKBest`, `RFE`, `feature_importances_`
- Viacnásobná regresia v scikit-learn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y_train)

model = LinearRegression()
model.fit(X_selected, y_train)
```

---

## 📚 Zdroje a odporúčania

- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [Hands-On ML with Scikit-Learn, Keras & TensorFlow (A. Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

## ✅ Odporúčania

- 📦 Použi `pip install scikit-learn pandas matplotlib` na inštaláciu
- 🧪 Pracuj v prostredí Jupyter Notebook (napr. VS Code, Google Colab)
- 💾 Používaj `.fit()` na učenie modelu a `.predict()` na predikciu

---

> Vytvoril: Miroslav Reiter (c) 2025  
> Pre kurz: [VITA Academy – Python ML kurz (začiatočník)](https://www.vita.sk)
