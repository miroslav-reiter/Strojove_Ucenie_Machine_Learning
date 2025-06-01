# 🤖 Online kurz: Strojové učenie (Machine Learning) v Pythone so scikit-learn

> Praktický kurz pre začiatočníkov – Regresia v ML, scikit-learn, modelovanie, tréning a vyhodnocovanie

---

## 📘 Obsah kurzu

01. [**🔍 Úvod do strojového učenia a regresie**](#uvod-ml-regresia))  
02. [**🧠 Prehľad typov regresii**](#prehlad-typov-regresii)
03. [**📦 Hlavné datasety v sklearn.datasets**](#prehlad-datasety)  
04. [**📈 Lineárna regresia v scikit-learn**](#linearna-regresia)  
05. [**🧮 Viacnásobná regresia a výber parametrov**](#viacnasobna-regresia)  
06. [**📚 Zdroje a literatúra k strojovemu uceniu a scikit-learn**](#zdroje-a-literatura)  
07. [**✅ Odporúčania ML, regresia a Scikit-Learn**](#odporucania)

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
<a name="prehlad-datasety"></a>
## 📦 2. Hlavné datasety v `sklearn.datasets`

> Prehľad zabudovaných datasetov v knižnici `scikit-learn`, rozdelený podľa typu úloh.

### 📦 Klasifikačné datasety (pre úlohy rozpoznávania kategórií)

| Dataset | Popis |
|--------|-------|
| `load_iris()` | 🪻 Iris – klasifikácia druhov kvetov na základe rozmerov okvetia. |
| `load_digits()` | 🔢 Rukou písané číslice (0–9) – obrázky 8x8 pixelov. |
| `load_wine()` | 🍷 Chemické vlastnosti vín – rozpoznanie odrody. |
| `load_breast_cancer()` | 🧪 Dáta o nádoroch prsníka – klasifikácia malígnych a benígnych. |
| `fetch_20newsgroups()` | 📰 Textové dáta z 20 kategórií diskusných skupín. |
| `fetch_20newsgroups_vectorized()` | 🔤 Predspracovaná verzia predchádzajúceho. |
| `fetch_covtype()` | 🌲 Lesné krytie – predikcia typu vegetácie na základe geografických znakov. |
| `fetch_kddcup99()` | 🌐 Sieťový traffic – detekcia anomálií a útokov. |
| `fetch_lfw_people()` | 👤 Rozpoznávanie osôb na obrázkoch (LFW). |
| `fetch_lfw_pairs()` | 👥 Porovnávanie tvárí – sú na dvoch obrázkoch rovnaké osoby? |
| `fetch_olivetti_faces()` | 🧑‍🦱 Dataset tvárí – rozpoznávanie identít. |
| `fetch_rcv1()` | 📰 Reuters texty – multilabel klasifikácia tém. |

### 📈 Regresné datasety

| Dataset | Popis |
|--------|-------|
| `load_diabetes()` | 🧬 Diabetes – predikcia progresie choroby. |
| `fetch_california_housing()` | 🏘️ Predikcia cien nehnuteľností v Kalifornii. |

### 💪 Iné a špeciálne datasety

| Dataset | Popis |
|--------|-------|
| `load_linnerud()` | 🏃‍♂️ Fyzické výkony a fyziologické dáta. |
| `fetch_species_distributions()` | 🐦 Výskyt druhov podľa geografie. |
| `load_files()` | 📂 Načítanie vlastných textových datasetov. |

### 🖼️ Datasety s obrázkami

| Dataset | Popis |
|--------|-------|
| `load_sample_image()` | 🖼️ Jednotlivý ukážkový obrázok (napr. čínska záhrada). |
| `load_sample_images()` | 🧩 Súbor ukážkových obrázkov. |

### ⚙️ Utility a nástroje

| Funkcia | Popis |
|--------|-------|
| `clear_data_home()` | 🧹 Vymaže cache dát scikit-learn. |
| `get_data_home()` | 📁 Získa cestu k dátovej zložke. |
| `fetch_openml()` | 🌐 Načítanie datasetov z OpenML. |
| `fetch_file()` | 📥 Stiahne súbor z webu do cache. |
| `load_svmlight_file()` | 📄 Načítanie SVMlight/libSVM formátu. |
| `load_svmlight_files()` | 📄 Viacero SVMlight súborov. |
| `dump_svmlight_file()` | 💾 Export dát do SVMlight. |

> Viac info: [sklearn.datasets API](https://scikit-learn.org/stable/api/sklearn.datasets.html#module-sklearn.datasets)

<a name="prehlad-typov-regresii"></a>
## 🧠 3. Prehľad typov regresií

Regresné modely sú určené na predpovedanie spojitých hodnôt. V tejto kapitole si predstavíme základné typy regresií, ich výhody, nevýhody a ukážeme si jednoduché príklady.

### 📊 Typy regresie:

| Typ regresie               | Popis | Príklad použitia |
|----------------------------|-------|------------------|
| Jednoduchá lineárna       | 1 vstupná premenná, lineárny vzťah | výška → hmotnosť |
| Viacnásobná lineárna      | Viac vstupných premenných | vek, BMI, príjem → krvný tlak |
| Polynomiálna regresia     | Obsahuje nelineárne členy (x², x³, ...) | vek² → výdavky |
| Ridge regresia            | Lineárna regresia s L2 regularizáciou | vysokodimenzionálne dáta |
| Lasso regresia            | Lineárna s L1 regularizáciou (výber premenných) | selekcia atribútov |
| ElasticNet                | Kombinácia L1 a L2 | kompromis medzi Ridge a Lasso |
| Logaritmická regresia     | Založená na logaritmickej transformácii | výskyt udalostí |
| Robustná regresia         | Odolná voči extrémnym hodnotám (outlierom) | analýza miezd |

---

### 🧪 Príklad: Polynomiálna regresia

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Používame len jeden atribút pre prehľadnosť
X_poly = X_train[['bmi']]
y_poly = y_train

# Vytvorenie modelu s polynómom 2. stupňa
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_poly, y_poly)

# Predikcia
X_test_poly = X_test[['bmi']]
y_pred_poly = poly_model.predict(X_test_poly)
```

---

### 🧪 Príklad: Ridge a Lasso regresia

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (L2 regularizácia)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
print("Ridge R2:", ridge_model.score(X_test, y_test))

# Lasso (L1 regularizácia)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
print("Lasso R2:", lasso_model.score(X_test, y_test))
```

➡️ Každý regresný model sa hodí na iný typ úlohy a dáta. Dôležité je analyzovať:
- linearitu vzťahu medzi premennými,
- počet a koreláciu vstupov,
- prítomnosť extrémnych hodnôt,
- a požiadavky na interpretáciu vs. výkon.

➡️ V ďalšej časti sa pozrieme na **lineárnu regresiu** v praxi – jej výpočet, vizualizáciu a interpretáciu.



<a name="linearna-regresia"></a>
## 📈 4. Lineárna regresia v scikit-learn

Lineárna regresia je základný model na predikciu spojitých hodnôt. Jej cieľom je nájsť optimálnu priamku (alebo hyperrovinu), ktorá minimalizuje chybu medzi predikovanými a skutočnými hodnotami.

### 📐 Rovnica lineárnej regresie

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$


kde $$(w_i \)$$  sú koeficienty modelu a $$( x_i \)$$  vstupné premenné.

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
## 🧮 5. Viacnásobná regresia a výber parametrov

Viacnásobná lineárna regresia rozširuje jednoduchú regresiu na viac vstupných premenných. Umožňuje lepšie modelovať komplexnejšie vzťahy v dátach.

### 📐 Rovnica viacnásobnej lineárnej regresie

$$
y = w₀ + w₁·x₁ + w₂·x₂ + … + wₙ·xₙ
$$

kde $$x₁...xₙ$$ sú vstupné atribúty a $$w₁...wₙ$$ ich koeficienty (váhy/weights).

$$
y = β0 + β1 x1 + β2 x2 + ⋯ + βn xn + ε
$$

kde $$x₁...xₙ$$ sú vstupné atribúty a $$w₁...wₙ$$ ich koeficienty (váhy/weights).

•  $$y$$ – závislá premenná (predikovaná hodnota)
•  $$x_1, x_2, ..., x_n$$ – nezávislé premenné (vysvetľujúce premenné)
•  $$β_0$$ – intercept (konštanta)
•  $$β_1, β_2, ..., β_n$$ – regresné koeficienty (váhy premenných)
•  $$ε$$ – náhodná chyba (reziduál)

Príklad pre 3 premenné:
$$
y = β0 + β1 x1 + β2 x2 + β3 x3 + ε
$$
Používa sa napríklad na predikciu cien, výnosov alebo skóre na základe viacerých vstupných faktorov.

---

### 🧪 Príklad 1: Tréning viacnásobného modelu so všetkými atribútmi

```python
from sklearn.linear_model import LinearRegression

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Predikcia
y_pred = model_multi.predict(X_test)

# Vyhodnotenie
from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

### 🧪 Príklad 2: Normalizácia vstupov pomocou StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

# Predikcia a vyhodnotenie
y_pred_scaled = model_scaled.predict(X_test_scaled)
print("R2 (škálované):", r2_score(y_test, y_pred_scaled))
```

---

### 🧪 Príklad 3: Výber najlepších atribútov pomocou SelectKBest

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Vyber 5 najrelevantnejších atribútov
selector = SelectKBest(score_func=f_regression, k=5)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

model_kbest = LinearRegression()
model_kbest.fit(X_train_selected, y_train)
print("R2 (SelectKBest):", model_kbest.score(X_test_selected, y_test))
```

➡️ Výber atribútov pomáha znížiť zložitosť modelu, odstrániť šum a zvýšiť interpretovateľnosť.
➡️ Normalizácia zaisťuje rovnaké váhové podmienky pre všetky atribúty.
➡️ Odporúčame testovať rôzne kombinácie atribútov a porovnávať metriky.


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

<a name="zdroje-a-literatura"></a>
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
<a name="odporucania"></a>
## ✅ Odporúčania ML, regresia a Scikit-Learn

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

