# 🤖 Online kurz: Strojové učenie (Machine Learning) v Pythone so scikit-learn

> Praktický kurz pre začiatočníkov – Regresia v ML, scikit-learn, modelovanie, tréning a vyhodnocovanie

---

## 📘 Obsah kurzu

01. [**🔍 Úvod do strojového učenia a regresie**](#uvod-ml-regresia)
01. [**📦 Hlavné datasety v sklearn.datasets**](#prehlad-datasety) 
01. [**🧠 Prehľad typov regresii**](#prehlad-typov-regresii)
01. [**📐 Regresné rovnice v strojovom učení**](#regresne-rovnice)
01. [**📈 Lineárna regresia v scikit-learn**](#linearna-regresia)  
01. [**🧮 Viacnásobná regresia a výber parametrov**](#viacnasobna-regresia)  
01. [**📚 Zdroje a literatúra k strojovemu uceniu a scikit-learn**](#zdroje-a-literatura)  
01. [**✅ Odporúčania ML, regresia a Scikit-Learn**](#odporucania)

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
#sklearn.show_versions()

# Verzia knižnice
print("scikit-learn verzia:", sklearn.__version__)

# Skontroluj dostupné moduly: linear_model, model_selection, metrics...
from sklearn import linear_model, metrics, preprocessing
print(dir(sklearn))
print(dir(linear_model))  # dostupné modely v linear_model
```

### 🧩 Prehľad hlavných modulov scikit-learn s kategorizáciou

| Modul                  | Popis                                                                 | Príklad použitia                                 | Kategória                        |
|------------------------|-----------------------------------------------------------------------|--------------------------------------------------|----------------------------------|
| `datasets`             | Vstavané datasety a generovanie syntetických údajov                   | `load_iris()`, `make_classification()`           | Základný modul                   |
| `model_selection`      | Rozdelenie dát, validácia, ladenie parametrov                         | `train_test_split()`, `GridSearchCV()`           | Základný modul                   |
| `metrics`              | Metriky pre hodnotenie modelov                                       | `accuracy_score()`, `r2_score()`                 | Základný modul                   |
| `linear_model`         | Lineárna a logistická regresia                                        | `LinearRegression()`, `LogisticRegression()`     | Základný modul                   |
| `tree`                 | Rozhodovacie stromy                                                   | `DecisionTreeClassifier()`                       | Základný modul                   |
| `ensemble`             | Zložené modely (RandomForest, Boosting...)                            | `RandomForestClassifier()`                       | Základný modul                   |
| `svm`                  | Support Vector Machines                                               | `SVC()`, `SVR()`                                 | Základný modul                   |
| `neighbors`            | Najbližší susedia                                                     | `KNeighborsClassifier()`                         | Základný modul                   |
| `naive_bayes`          | Naívne Bayesove klasifikátory                                         | `GaussianNB()`                                   | Základný modul                   |
| `neural_network`       | Viacvrstvové neurónové siete                                          | `MLPClassifier()`, `MLPRegressor()`              | Základný modul                   |
| `preprocessing`        | Úprava dát: škálovanie, normalizácia, kódovanie                       | `StandardScaler()`, `OneHotEncoder()`            | Základný modul                   |
| `impute`               | Doplňovanie chýbajúcich hodnôt                                       | `SimpleImputer()`, `KNNImputer()`                | Základný modul                   |
| `pipeline`             | Zreťazenie krokov (transformácie + model)                             | `Pipeline([...])`                                | Základný modul                   |
| `feature_selection`    | Výber najdôležitejších vlastností                                     | `SelectKBest()`, `RFE()`                         | Základný modul                   |
| `feature_extraction`   | Extrakcia znakov z textu, obrázkov                                    | `CountVectorizer()`, `TfidfTransformer()`        | Základný modul                   |
| `decomposition`        | Redukcia dimenzie (napr. PCA)                                         | `PCA()`, `TruncatedSVD()`                        | Pokročilejší modul              |
| `manifold`             | Nelineárna redukcia dimenzie                                          | `TSNE()`, `Isomap()`                             | Pokročilejší modul              |
| `cluster`              | Klastrovanie bez dozorovania                                          | `KMeans()`, `DBSCAN()`                           | Pokročilejší modul              |
| `mixture`              | Zmesové modely (pravdepodobnostné klastrovanie)                      | `GaussianMixture()`                              | Pokročilejší modul              |
| `discriminant_analysis`| LDA a QDA pre viac tried                                              | `LinearDiscriminantAnalysis()`                   | Pokročilejší modul              |
| `multiclass`           | Rozšírenia pre viac ako 2 tried                                       | `OneVsRestClassifier()`                          | Pokročilejší modul              |
| `multioutput`          | Modely s viacerými výstupmi                                           | `MultiOutputClassifier()`                        | Pokročilejší modul              |
| `experimental`         | Funkcie v experimentálnom stave                                       | `HistGradientBoostingClassifier()`               | Experimentálny modul            |
| `inspection`           | Interpretácia modelov                                                 | `permutation_importance()`                       | Pokročilejší modul              |
| `compose`              | Kombinovanie transformácií                                            | `ColumnTransformer()`                            | Pokročilejší modul              |
| `random_projection`    | Redukcia dimenzie náhodnou projekciou                                | `GaussianRandomProjection()`                     | Pokročilejší modul              |
| `gaussian_process`     | Modely založené na Gaussových procesoch                               | `GaussianProcessRegressor()`                     | Pokročilejší modul              |
| `isotonic`             | Izotonická (monotónna) regresia                                       | `IsotonicRegression()`                           | Pokročilejší modul              |
| `kernel_approximation` | Približné jadrové transformácie                                       | `RBFSampler()`                                   | Pokročilejší modul              |
| `kernel_ridge`         | Kombinácia ridge a kernel metód                                      | `KernelRidge()`                                  | Pokročilejší modul              |
| `externals`            | Interné závislosti (napr. `joblib`)                                   | –                                                | Podporný modul                  |
| `exceptions`           | Definície chýb a výnimiek                                             | –                                                | Podporný modul                  |
| `get_config`           | Získanie globálnej konfigurácie                                       | `get_config()`                                   | Konfigurácia a nástroje         |
| `set_config`           | Nastavenie globálnej konfigurácie                                     | `set_config(display='diagram')`                  | Konfigurácia a nástroje         |
| `config_context`       | Dočasná zmena konfigurácie                                            | `with config_context():`                         | Konfigurácia a nástroje         |
| `show_versions`        | Výpis verzií knižnice a závislostí                                    | `show_versions()`                                | Konfigurácia a nástroje         |
| `clone`                | Kopírovanie modelov                                                   | `clone(model)`                                   | Konfigurácia a nástroje         |


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


<a name="regresne-rovnice"></a>
## 📐 4. Regresné rovnice v strojovom učení

### 1️⃣ Lineárna regresia (Simple Linear Regression)
**Rovnica:**  
`y = β₀ + β₁·x + ε`

- `y` – predikovaná hodnota (napr. cena)
- `x` – nezávislá premenná (napr. rozloha)
- `β₀` – intercept (konštanta, keď x = 0)
- `β₁` – koeficient sklonu (ako rýchlo y rastie s x)
- `ε` – chyba modelu (residuum)

✔ Príklad: Predikcia ceny domu na základe výmery.  
✔ Príklad: Odhad spotreby energie podľa vonkajšej teploty.

### 2️⃣ Viacnásobná lineárna regresia (Multiple Linear Regression)
**Rovnica:**  
`y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε`

- Používa viacero vstupných premenných (napr. príjem, lokalita, vek budovy)
- Koeficienty `β₁...βₙ` vyjadrujú význam každého vstupu

✔ Príklad: Predikcia ceny auta podľa veku, značky, výkonu a najazdených km.  
✔ Príklad: Odhad výsledkov študenta podľa dochádzky, času učenia a spánku.

### 3️⃣ Polynomiálna regresia (Polynomial Regression)
**Rovnica:**  
`y = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ + ε`

- Rozširuje lineárnu regresiu o vyššie mocniny `x`
- Modeluje nelineárne závislosti

✔ Príklad: Predikcia výšky rastliny v závislosti od času (krivka rastu).  
✔ Príklad: Odhad výšky skoku v závislosti od rýchlosti a uhla odrazu.

### 4️⃣ Ridge regresia (L2 regularizácia)
**Rovnica:**  
`minimize: ||y - Xβ||² + λ·||β||²`

- Penalizuje veľké koeficienty pomocou L2 normy
- Pomáha pri multikolinearite a znižuje pretrénovanie

✔ Vhodná ak máte mnoho podobných vstupných premenných.  
✔ Príklad: Predikcia nákladov na marketing z desiatok prepojených metrík.

### 5️⃣ Lasso regresia (L1 regularizácia)
**Rovnica:**  
`minimize: ||y - Xβ||² + λ·|β|`

- Penalizuje súčet absolútnych hodnôt koeficientov
- Pomáha automaticky vyberať dôležité premenné (niektoré β = 0)

✔ Užitočné pri veľkom počte premenných a potrebe výberu.  
✔ Príklad: Výber najvplyvnejších faktorov ovplyvňujúcich cenu nehnuteľnosti.

### 6️⃣ Elastic Net
**Rovnica:**  
`minimize: ||y - Xβ||² + λ₁·|β| + λ₂·||β||²`

- Kombinácia Ridge a Lasso
- Lepšia stabilita pri vysoko korelovaných premenných

✔ Zvyčajne sa nastavujú váhy (α, λ) cez cross-validation.  
✔ Príklad: Predikcia predaja produktov pri vysoko závislých marketingových faktoroch.

### 7️⃣ Logaritmická regresia (pre log-transformed y)
**Rovnica:**  
`ln(y) = β₀ + β₁·x + ε`

- Používa sa ak cieľová premenná má exponenciálne rozdelenie
- Výstup sa často exponuje späť: `y = e^(predikcia)`

✔ Príklad: Predikcia populácie alebo ceny pri log-normálnom rozdelení.  
✔ Príklad: Odhad rastu používateľov novej aplikácie.

### 8️⃣ Regresia pomocou rozhodovacích stromov
**Princíp:**  
Nie je založená na rovniciach, ale na rekurzívnom delení:

```python
if x₁ < 2.5: predikcia = 100
elif x₂ > 5: predikcia = 200
```

✔ Výhoda: nepotrebuje škálovanie dát ani lineárne vzťahy.  
✔ Príklad: Predikcia výdavkov domácnosti na základe segmentácie podľa veku a príjmu.

### 9️⃣ Support Vector Regression (SVR)
**Princíp:**  
Nájsť funkciu `f(x)`, ktorá sa líši od `y` o maximálne `ε` a je čo najplochšia.

- Umožňuje nelineárne vzťahy pomocou kernelov (napr. RBF)
- Funguje dobre aj pri vysokorozmerných dátach

✔ Príklad: Predikcia cien akcií s využitím RBF kernelu.  
✔ Príklad: Odhaľovanie trendov v meteorologických dátach.

### 🔟 K-nearest neighbors regresia (KNN Regression)
**Rovnica:**  
`y_pred = priemer(y susedov)`

- Predikcia = priemer výstupov najbližších `K` bodov
- Funguje na základe vzdialenosti medzi dátami

✔ Neparametrický, jednoduchý model.  
✔ Príklad: Odhad ceny Airbnb na základe okolitých ponúk.  
✔ Príklad: Predikcia času dochádzky podľa najbližších historických dát.  
✔ Príklad: Odhad návštevnosti podujatia na základe podobných predchádzajúcich akcií.

| Typ regresie                    | Rovnica / princíp                    | Typ modelu                          | Použitie                           | Príklad                                                                                  |
|:--------------------------------|:-------------------------------------|:------------------------------------|:-----------------------------------|:-----------------------------------------------------------------------------------------|
| Lineárna regresia               | y = β₀ + β₁·x + ε                    | Parametrický                        | Jednoduché lineárne vzťahy         | Cena domu podľa výmery, mzda podľa odpracovaných hodín, dopyt podľa ceny                 |
| Viacnásobná lineárna regresia   | y = β₀ + β₁·x₁ + β₂·x₂ + ... + ε     | Parametrický                        | Viacero vstupov                    | Cena auta podľa veku, značky, výkonu, najazdených km, spotreby                           |
| Polynomiálna regresia           | y = β₀ + β₁·x + β₂·x² + ... + ε      | Parametrický                        | Nelineárne vzťahy                  | Rast rastliny v čase, zmena teploty počas dňa, pokles výkonu batérie                     |
| Ridge regresia (L2)             | min ||y - Xβ||² + λ·||β||²           | Parametrický (regularizovaný)       | Multikolinearita                   | Náklady na marketing, výkon modelu podľa počtu iterácií, výdavky podľa kategórií         |
| Lasso regresia (L1)             | min ||y - Xβ||² + λ·|β|              | Parametrický (s výberom premenných) | Výber relevantných vstupov         | Predikcia ceny nehnuteľnosti, výber kľúčových premenných, zjednodušenie modelu           |
| Elastic Net                     | min ||y - Xβ||² + λ₁·|β| + λ₂·||β||² | Parametrický (kombinovaný)          | Korelované premenné                | Predaj produktu z marketingových metrík, odhad výnosnosti reklamy, optimalizácia kampaní |
| Logaritmická regresia           | ln(y) = β₀ + β₁·x + ε                | Parametrický (log-transformovaný)   | Log-normálne rozdelenie            | Predikcia populácie, rast cien v čase, počet používateľov aplikácie                      |
| Rozhodovacie stromy             | Podmienkové delenie                  | Neparametrický                      | Segmentácia a rozhodovanie         | Výdavky domácnosti, odhad ceny podľa kategórií, správanie zákazníkov                     |
| Support Vector Regression (SVR) | ε-insensitive loss + kernel          | Parametrický / Kernelový            | Nelineárne vzťahy, vysoká dimenzia | Predikcia cien akcií, komplexné vzťahy medzi trhmi, analýza trendov                      |
| K-nearest neighbors (KNN)       | y_pred = priemer(y susedov)          | Neparametrický                      | Podobnosť podľa vzdialenosti       | Cena Airbnb podľa okolia, odporúčanie produktov, odhad nákladov na základe podobností    |


<a name="linearna-regresia"></a>
## 📈 4. Lineárna regresia v scikit-learn

Lineárna regresia je základný model na predikciu spojitých hodnôt. Jej cieľom je nájsť optimálnu priamku (alebo hyperrovinu), ktorá minimalizuje chybu medzi predikovanými a skutočnými hodnotami.

### 📐 Rovnica lineárnej regresie  

**Strojové učenie / ML notácia:** (používaná v machine learning, neurónových sieťach)  

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n  
$$

kde $$(w_i \)$$  sú koeficienty modelu a $$( x_i \)$$  vstupné premenné.  

**Matematická a štatistická notácia:** (používaná v štatistike, ekonometrii, vede)

$$ 
y = β0 + β1.x1 + β2.x2 + ⋯ + βn.xn + ε  
$$

Kde:  
•	$$y$$ – závislá premenná  
•	$$x$$ – nezávislá premenná  
•	$$β₀$$ – intercept (konštanta)  
•	$$β₁$$ – smernica (koeficient regresie)  
•	$$ε$$ – náhodná chyba  


---

### 📦 Prehľad modelov v `sklearn.linear_model`

| Trieda / Model                        | Popis                                                                                 | Typ úlohy       | Poznámka / Vlastnosť                     |
|--------------------------------------|----------------------------------------------------------------------------------------|------------------|-------------------------------------------|
| `LinearRegression`                   | Obyčajná lineárna regresia (metóda najmenších štvorcov)                               | Regresia         | Bez regularizácie                         |
| `Ridge`                              | Lineárna regresia s L2 regularizáciou                                                  | Regresia         | Trestá veľké koeficienty                 |
| `Lasso`                              | Lineárna regresia s L1 regularizáciou                                                  | Regresia         | Môže úplne vynulovať niektoré koeficienty |
| `ElasticNet`                         | Kombinácia L1 a L2 regularizácie                                                       | Regresia         | Flexibilné nastavenie penalizácie        |
| `RidgeCV`                            | Ridge s automatickým výberom parametra cez cross-validáciu                            | Regresia         | Interná validácia                        |
| `LassoCV`                            | Lasso s výberom optimálneho α                                                          | Regresia         | Automatické ladenie                      |
| `ElasticNetCV`                       | ElasticNet s výberom parametrov cez cross-validáciu                                    | Regresia         | Automatický tuning                       |
| `Lars`                               | Least Angle Regression – efektívna pre veľa premenných                                 | Regresia         | Alternatíva k Lasso                      |
| `LarsCV`                             | Lars s výberom cez cross-validáciu                                                     | Regresia         |                                          |
| `LassoLars`                          | Lasso implementované pomocou LARS algoritmu                                            | Regresia         | Rýchle pri veľkom počte vstupov          |
| `LassoLarsCV`                        | Verzia s výberom parametra cez cross-validáciu                                        | Regresia         |                                          |
| `LassoLarsIC`                        | Výber modelu cez AIC / BIC                                                            | Regresia         | Informačné kritériá                      |
| `OrthogonalMatchingPursuit`         | Greedy algoritmus pre riedku regresiu                                                  | Regresia         | Pre riedke (sparse) modely               |
| `OrthogonalMatchingPursuitCV`       | Verzia s výberom počtu koeficientov                                                    | Regresia         |                                          |
| `HuberRegressor`                    | Robustná regresia odolná voči odľahlým hodnotám                                        | Regresia         | Kombinuje vlastnosti Ridge a L1          |
| `RANSACRegressor`                   | Detekcia odľahlých hodnôt cez iteratívny výber vzoriek                                | Regresia         | Odolný voči šumu                         |
| `TheilSenRegressor`                 | Robustná štatistická regresia                                                         | Regresia         | Pomalejší, ale stabilný pri outlieroch   |
| `BayesianRidge`                     | Bayesovská lineárna regresia                                                          | Regresia         | Pravdepodobnostný prístup                |
| `ARDRegression`                     | Automatické určenie relevantných premenných                                           | Regresia         | Automatické vynulovanie nerelevantných   |
| `PoissonRegressor`                 | Poissonova regresia pre početné dáta                                                  | Regresia         | Generalizovaný lineárny model (GLM)      |
| `GammaRegressor`                    | Regressia s gama rozdelením                                                           | Regresia         | GLM pre kladné spojité hodnoty           |
| `TweedieRegressor`                  | Generalizovaný lineárny model pre rôzne distribúcie                                   | Regresia         | Flexibilné nastavenie parametra          |
| `QuantileRegressor`                 | Kvantilová regresia – predikcia percentilu (nie priemeru)                             | Regresia         | Vhodné pre odhad intervalov              |
| `LogisticRegression`                | Binárna alebo viactriedna klasifikácia                                                 | Klasifikácia     | Najčastejšie používaný lineárny klasifikátor |
| `LogisticRegressionCV`             | LogisticRegression s výberom parametra cez cross-validáciu                            | Klasifikácia     | Automatické ladenie                      |
| `Perceptron`                        | Jednoduchý neurón pre binárnu klasifikáciu                                            | Klasifikácia     | Bez skrytej vrstvy                       |
| `PassiveAggressiveClassifier`       | Efektívny algoritmus pre veľké datasety (online učenie)                               | Klasifikácia     | Vhodné pre streamovanie dát              |
| `PassiveAggressiveRegressor`        | Variant pre regresiu                                                                 | Regresia         | Online učenie                            |
| `SGDClassifier`                     | Klasifikácia pomocou stochastického gradientu                                         | Klasifikácia     | Veľmi rýchly, online učenie              |
| `SGDRegressor`                      | Regresia pomocou SGD                                                                  | Regresia         | Veľké datasety, regularizácia            |
| `SGDOneClassSVM`                    | Jednotriedna detekcia anomálií pomocou SGD                                            | Anomálie         | Experimentálne                           |

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

**Strojové učenie / ML notácia:** (používaná v machine learning, neurónových sieťach)    

$$  
y = w₀ + w₁·x₁ + w₂·x₂ + … + wₙ·xₙ  
$$  

kde $$x₁...xₙ$$ sú vstupné atribúty a $$w₁...wₙ$$ ich koeficienty (váhy/weights), často sa nepíše ε (predikcia bez explicitnej chyby).

**Matematická a štatistická notácia:** (používaná v štatistike, ekonometrii, vede)  

$$  
y = β₀ + β₁ x₁ + β2 x₂ + ⋯ + βₙ xₙ + ε  
$$  

kde $$x₁...xₙ$$ sú vstupné atribúty a $$w₁...wₙ$$ ich koeficienty (váhy/weights).

•  $$y$$ – závislá premenná (predikovaná hodnota)  
•  $$x₁, x₂, ..., xₙ$$ – nezávislé premenné (vysvetľujúce premenné)  
•  $$β₀$$ – intercept (konštanta)  
•  $$β₁, β₂, ..., βₙ$$ – regresné koeficienty (váhy premenných)  
•  $$ε$$ – náhodná chyba (reziduál)  

Príklad pre 3 premenné:  

$$
y = β0 + β1.x1 + β2.x2 + β3.x3 + ε
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
➡️ Odporúčam testovať rôzne kombinácie atribútov a porovnávať metriky.


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

