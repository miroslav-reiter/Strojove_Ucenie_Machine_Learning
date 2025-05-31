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

# Načítanie vstupného datasetu diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Rozdelenie na tréningovú a testovaciu množinu (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Výpis tvaru datasetu
print(X_train.shape, X_test.shape)
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
import matplotlib.pyplot as plt

# Tréning modelu na jednom atribúte (napr. BMI)
model = LinearRegression()
model.fit(X_train[['bmi']], y_train)

# Predikcia hodnôt
preds = model.predict(X_test[['bmi']])

# Vyhodnotenie výkonu modelu
print("R2 score:", r2_score(y_test, preds))
print("Mean Squared Error:", mean_squared_error(y_test, preds))

# Vizualizácia
plt.scatter(X_test['bmi'], y_test, color='blue', label='Skutočné hodnoty')
plt.plot(X_test['bmi'], preds, color='red', label='Predikované hodnoty')
plt.xlabel('BMI')
plt.ylabel('Cieľová hodnota')
plt.title('Lineárna regresia na BMI')
plt.legend()
plt.show()
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

# Normalizácia vstupných údajov
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Výber 5 najlepších atribútov podľa korelácie s cieľovou premennou
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y_train)

# Tréning viacnásobného regresného modelu
model = LinearRegression()
model.fit(X_selected, y_train)

# Vyhodnotenie na testovacích dátach
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

preds = model.predict(X_test_selected)
print("R2 score:", r2_score(y_test, preds))
print("Mean Squared Error:", mean_squared_error(y_test, preds))
```

---

## 📚 Zdroje a literatúra k strojovému učeniu a scikit-learn

### 📘 Top knihy – Anglicky

| Názov | Autor | Popis |
|-------|-------|-------|
| Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow | Aurélien Géron | Najkomplexnejšia kniha pre ML v Pythone. Obsahuje teóriu aj praktické príklady. |
| Python Machine Learning | Sebastian Raschka, Vahid Mirjalili | Kniha zameraná na praktickú implementáciu ML algoritmov v Pythone. |
| Introduction to Machine Learning with Python | Andreas Müller, Sarah Guido | Výborný úvod do ML so zameraním na scikit-learn. |
| Machine Learning Yearning | Andrew Ng | Strategický pohľad na budovanie ML systémov (voľne dostupné PDF). |
| Pattern Recognition and Machine Learning | Christopher Bishop | Hlbšie teoretické základy ML. |
| Data Science from Scratch | Joel Grus | Základy ML s implementáciou algoritmov "od nuly". |

### 📙 Knihy a zdroje – Česky a Slovensky

| Názov | Autor | Popis |
|-------|-------|-------|
| Učíme se strojové učení | Ondřej Vaňo | Česká príručka s praktickými úlohami v Pythone. |
| Základy strojového učenia | kol. autorov | Úvod do ML a základné modely v češtine. |
| Python pro analýzu dat | Wes McKinney (CZ preklad) | Zamerané na prácu s dátami v Pandas, vhodné ako základ pre ML. |
| Strojové učenie v Pythone | Miroslav Reiter | Praktický kurz pre začiatočníkov s dôrazom na regresiu a scikit-learn. |

> 🧠 Poznámka: Československé tituly sú často pre začiatočníkov a pomáhajú pochopiť základy v rodnom jazyku.

### 🌐 Oficiálna dokumentácia a online zdroje

| Názov | Popis | Odkaz |
|-------|--------|--------|
| scikit-learn User Guide | Oficiálna dokumentácia a príručky | [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html) |
| scikit-learn API Reference | Detailné API popisy tried a funkcií | [scikit-learn.org API](https://scikit-learn.org/stable/modules/classes.html) |
| sklearn-examples GitHub | Príklady projektov s použitím scikit-learn | [github.com/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| Kaggle Learn ML | Online micro-kurzy a súťaže | [kaggle.com/learn](https://www.kaggle.com/learn/intro-to-machine-learning) |
| Machine Learning Mastery | Blog a návody pre praktické ML | [machinelearningmastery.com](https://machinelearningmastery.com) |
| Google AI Hub | Nástroje, datasetové repozitáre a príklady | [ai.google](https://ai.google/tools/) |
| Towards Data Science | Blogy a tutoriály od expertov z ML komunity | [towardsdatascience.com](https://towardsdatascience.com) |

---

## ✅ Odporúčania

- 📦 Použi `pip install scikit-learn pandas matplotlib` na inštaláciu
- 🧪 Pracuj v prostredí Jupyter Notebook (napr. Jetbrains Datalore, VS Code, Google Colab)
- 💾 Používaj `.fit()` na učenie modelu a `.predict()` na predikciu nových vstupov
- 🔍 Využívaj `train_test_split()` na oddelenie tréningových a testovacích dát
- 🛠 Skúšaj rôzne metriky výkonu (napr. R², MSE, MAE) podľa typu úlohy
- 📊 Vizualizuj výsledky cez `matplotlib` alebo `seaborn` pre lepšie porozumenie dát
- 🧠 Pri výbere atribútov použi `SelectKBest`, `RFE` alebo `feature_importances_`
- 🔁 Používaj `cross_val_score()` na krížovú validáciu modelu
- 🧪 Nezabudni na `StandardScaler()` alebo `MinMaxScaler()` pri škálovaní údajov
- 🗃️ Na prácu s väčšími datasetmi vyskúšaj `fetch_california_housing` alebo vlastné CSV súbory
- 🧾 Sleduj novinky na [scikit-learn release log](https://scikit-learn.org/stable/whats_new.html) pre nové funkcie a zmeny

---

