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

V tejto úvodnej časti sa zoznámime so základnými pojmami, rozdielmi medzi typmi ML úloh, knižnicou scikit-learn a jednoduchými praktickými príkladmi.

### 🎯 Základné pojmy

- **Supervised learning (učenie s učiteľom)** – algoritmus sa učí na základe označených dát (X vstupy, y výstupy)
- **Training (tréning)** – fáza učenia modelu na základe historických údajov
- **Testing (testovanie)** – overenie výkonu modelu na nových údajoch, ktoré nevidel
- **Regresia vs. klasifikácia**:
  - *Regresia* predpovedá **spojité hodnoty** (napr. cena, teplota)
  - *Klasifikácia* predpovedá **kategórie** (napr. áno/nie, trieda A/B/C)

### 🧩 Typy regresií (prehľad)

- **Jednoduchá lineárna regresia** – 1 vstupná premenná (napr. výška → hmotnosť)
- **Viacnásobná lineárna regresia** – viac vstupov (napr. výška, vek, BMI → hmotnosť)
- **Polynomiálna regresia** – rozšírenie lineárnej pomocou nelineárnych zložiek

---

### 🧪 Príklad 1: Načítanie dát a základná štatistika

```python
from sklearn.datasets import load_diabetes
import pandas as pd

# Načítanie datasetu (zabudovaný dataset s údajmi o cukrovke)
data = load_diabetes(as_frame=True)
df = data.frame

# Zobrazenie prvých 5 riadkov
print(df.head())

# Základná štatistika
print(df.describe())
```

---

### 🧪 Príklad 2: Rozdelenie dát na tréningovú a testovaciu množinu

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='target')  # vstupy (atribúty)
y = df['target']               # cieľová premenná

# Rozdelenie na 80 % tréning a 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tréningová množina:", X_train.shape)
print("Testovacia množina:", X_test.shape)
```

---

### 🧪 Príklad 3: Základná vizualizácia vzťahu medzi atribútom a cieľom

```python
import matplotlib.pyplot as plt

# Porovnanie BMI a cieľovej hodnoty
plt.scatter(X['bmi'], y, color='green', alpha=0.5)
plt.xlabel('BMI')
plt.ylabel('Cieľová premenná (target)')
plt.title('Vzťah medzi BMI a cieľom')
plt.grid(True)
plt.show()
```

---

### 🧪 Príklad 4: Prehľad funkcií knižnice scikit-learn

```python
import sklearn

# Verzia knižnice
print("scikit-learn verzia:", sklearn.__version__)

# Skontroluj dostupné moduly: linear_model, model_selection, metrics...
from sklearn import linear_model, metrics, preprocessing
print(dir(linear_model))  # dostupné modely v linear_model
```

Týmto sme získali základný prehľad o tom:
- ako vyzerajú dáta,
- ako sa rozdeľujú na tréning a test,
- ako vizualizovať vzťahy a
- čo ponúka knižnica scikit-learn.

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

Lineárna regresia je základný model na predikciu spojitých hodnôt. Jej cieľom je nájsť optimálnu priamku (alebo hyperrovinu), ktorá minimalizuje chybu medzi predikovanými a skutočnými hodnotami.

### 📐 Rovnica lineárnej regresie

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$


kde $$w_i \)$$  sú koeficienty modelu a $$( x_i \)$$  vstupné premenné.

---

### 🧪 Príklad 1: Jednoduchá lineárna regresia (1 vstup)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Vytvorenie modelu a natrénovanie na jednom atribúte (napr. BMI)
model = LinearRegression()
model.fit(X_train[['bmi']], y_train)

# Predikcia hodnôt na testovacej množine
preds = model.predict(X_test[['bmi']])

# Vyhodnotenie modelu
print("R2 score:", r2_score(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
```

---

### 🧪 Príklad 2: Vizualizácia regresnej priamky

```python
import matplotlib.pyplot as plt

# Skutočné vs. predikované hodnoty
plt.scatter(X_test['bmi'], y_test, color='gray', label='Skutočné')
plt.plot(X_test['bmi'], preds, color='red', label='Predikované')
plt.xlabel('BMI')
plt.ylabel('Cieľová premenná')
plt.title('Lineárna regresia: BMI vs. cieľ')
plt.legend()
plt.show()
```

---

### 🧪 Príklad 3: Zobrazenie koeficientov modelu

```python
# Koeficient (sklon priamky)
print("Koeficient w1:", model.coef_[0])

# Absolútna hodnota intercept (posun priamky)
print("Intercept w0:", model.intercept_)
```

---

### 🧪 Príklad 4: Viac atribútov naraz – multivariačná lineárna regresia

```python
# Všetky atribúty použité na tréning
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Predikcia a vyhodnotenie
preds_multi = model_multi.predict(X_test)
print("R2 (viac atribútov):", r2_score(y_test, preds_multi))
```


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

