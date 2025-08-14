export const codeGenSystem = (project, dataAnalysis, hyperparamsSuggestion) => `
# Dieser Code ist ein generisches, ausführbares ML-Skript, das anhand der im Prompt
# übergebenen Projekt-Parameter (${JSON.stringify(project)}) eine robuste Pipeline aufsetzt.
# Es lädt Daten (CSV/JSON/Excel), bereinigt, preprocessiert (Skalierung/Encoding),
# führt Train/Test-Split durch, trainiert das angegebene Modell mit bereitgestellten
# Hyperparametern und einer optionalen, automatischen Feinabstimmung um die Vorgaben herum,
# gibt mehrere Metriken aus und speichert Modell sowie Label-Encoder.

# WICHTIG:
# - Dateipfad MUSS r"${project.csvFilePath}" sein.
# - hyperparameters MUSS in main() als JSON-String zugewiesen werden:
# hyperparameters = "${JSON.stringify(project.hyperparameters).replace(/"/g, '\\"')}"
# - Dieses Skript nutzt scikit-learn Pipelines und kann XGBoost unterstützen (falls installiert).
# - Alle Ausgaben sind so formatiert, dass sie maschinell parsebar sind (z.B. "Accuracy: 0.8524").

# Template-Code:
import json
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model\_selection import train\_test\_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
accuracy\_score,
precision\_score,
recall\_score,
f1\_score,
roc\_auc\_score,
confusion\_matrix,
classification\_report,
r2\_score,
mean\_squared\_error,
mean\_absolute\_error,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear\_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

try:
from xgboost import XGBClassifier, XGBRegressor  # type: ignore
\_XGB\_AVAILABLE = True
except Exception:
\_XGB\_AVAILABLE = False

import joblib

def log(msg: str) -> None:
print(msg, flush=True)

def load\_source(path: str) -> pd.DataFrame:
ext = os.path.splitext(path)\[1].lower()
if ext == ".csv":
return pd.read\_csv(path)
if ext == ".json":
return pd.read\_json(path, lines=False)
if ext in (".xlsx", ".xls"):
return pd.read\_excel(path)
raise ValueError(f"Unsupported file extension: {ext}")

def try\_parse\_datetime(df: pd.DataFrame, obj\_cols: List\[str]) -> pd.DataFrame:
for c in obj\_cols:
s = df\[c]
if s.isna().all():
continue
sample = s.dropna().astype(str).head(50)
\# Heuristik: wenn mind. 30% der Werte als Datum parsebar, dann zu datetime konvertieren
parsed = 0
for v in sample:
try:
pd.to\_datetime(v, errors="raise")
parsed += 1
except Exception:
continue
if len(sample) > 0 and parsed / len(sample) >= 0.3:
df\[c] = pd.to\_datetime(df\[c], errors="coerce")
return df

def datetime\_feature\_engineering(df: pd.DataFrame) -> Tuple\[pd.DataFrame, List\[str]]:
new\_cols: List\[str] = \[]
for c in df.columns:
if np.issubdtype(df\[c].dtype, np.datetime64):
df\[f"{c}\_\_year"] = df\[c].dt.year
df\[f"{c}\_\_month"] = df\[c].dt.month
df\[f"{c}\_\_day"] = df\[c].dt.day
df\[f"{c}\_\_weekday"] = df\[c].dt.weekday
df\[f"{c}\_\_hour"] = df\[c].dt.hour
new\_cols += \[f"{c}\_\_year", f"{c}\_\_month", f"{c}\_\_day", f"{c}\_\_weekday", f"{c}\_\_hour"]
\# Original-Datetime-Spalte entfernen (Encoder/Scaler-unfreundlich)
df.drop(columns=\[c], inplace=True)
return df, new\_cols

def basic\_cleaning(df: pd.DataFrame) -> pd.DataFrame:
before = len(df)
df = df.drop\_duplicates().copy()
after = len(df)
log(f"INFO: DroppedDuplicates: {before - after}")
\# Trim strings
obj\_cols = df.select\_dtypes(include=\["object"]).columns.tolist()
for c in obj\_cols:
try:
df\[c] = df\[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
except Exception:
pass
\# Parse mögliche Datumsfelder
df = try\_parse\_datetime(df, obj\_cols)
\# Feature Engineering auf Datumsfelder
df, \_ = datetime\_feature\_engineering(df)
return df

def infer\_problem\_type(model\_type: Optional\[str], y: pd.Series) -> str:
if model\_type and model\_type.lower() in ("classification", "regression"):
return model\_type.lower()
\# Heuristik: diskrete, wenige Klassen -> Classification
if y.dtype.kind in ("i", "b") and y.nunique() <= max(20, int(0.05 \* len(y))):
return "classification"
if y.dtype.kind == "O":
return "classification"
return "regression"

def coerce\_numeric\_json\_values(params: Dict\[str, Any]) -> Dict\[str, Any]:
def coerce(v: Any) -> Any:
if isinstance(v, str):
try:
if v.lower() in ("true", "false"):
return v.lower() == "true"
except Exception:
pass
try:
if "." in v or "e" in v.lower():
return float(v)
return int(v)
except Exception:
return v
if isinstance(v, list):
return \[coerce(x) for x in v]
if isinstance(v, dict):
return {k: coerce(val) for k, val in v.items()}
return v
return {k: coerce(val) for k, val in params.items()}

def build\_estimator(algorithm: str, problem\_type: str, params: Dict\[str, Any]):
algo = algorithm.strip()
if problem\_type == "classification":
if algo == "RandomForestClassifier":
return RandomForestClassifier(\*\*params)
if algo == "LogisticRegression":
return LogisticRegression(max\_iter=1000, \*\*params)
if algo == "SVM":
return SVC(probability=True, \*\*params)
if algo == "XGBoostClassifier":
if not \_XGB\_AVAILABLE:
raise ImportError("XGBoost not available. Please install xgboost.")
return XGBClassifier(\*\*params)
else:
if algo == "RandomForestRegressor":
return RandomForestRegressor(\*\*params)
if algo == "LinearRegression":
return LinearRegression(\*\*params)
if algo == "SVR":
return SVR(\*\*params)
if algo == "XGBoostRegressor":
if not \_XGB\_AVAILABLE:
raise ImportError("XGBoost not available. Please install xgboost.")
return XGBRegressor(\*\*params)
raise ValueError(f"Unsupported algorithm '{algorithm}' for problem type '{problem\_type}'")

def build\_search\_space(algorithm: str, problem\_type: str, base: Dict\[str, Any], n\_features: int, n\_samples: int) -> Dict\[str, Any]:
\# Heuristische Suchräume um gegebene Hyperparameter (oder sinnvolle Defaults) herum
space: Dict\[str, Any] = {}
def around\_int(key: str, default: int, low: int = 1, high: Optional\[int] = None, scale: float = 0.5):
v = int(base.get(key, default))
delta = max(1, int(v \* scale))
hi = v + delta if high is None else min(high, v + delta)
lo = max(low, v - delta)
space\[key] = list(sorted(set(\[lo, v, hi])))

def around_float(key: str, default: float, scale: float = 0.5, low: Optional[float] = None, high: Optional[float] = None):
    v = float(base.get(key, default))
    lo = v * (1 - scale)
    hi = v * (1 + scale)
    if low is not None:
        lo = max(lo, low)
    if high is not None:
        hi = min(hi, high)
    space[key] = [lo, v, hi]

algo = algorithm.strip()
if problem_type == "classification":
    if algo == "RandomForestClassifier":
        around_int("n_estimators", default=200, low=50, high=1000, scale=1.0)
        around_int("max_depth", default= None if base.get("max_depth") in (None, "None") else int(base.get("max_depth", 20)), low=3, high=64, scale=1.0)
        around_int("min_samples_split", default=2, low=2, high=20, scale=1.0)
        around_int("min_samples_leaf", default=1, low=1, high=10, scale=1.0)
        space.setdefault("max_features", ["sqrt", "log2", None])
        space.setdefault("bootstrap", [True, False])
    elif algo == "LogisticRegression":
        around_float("C", default=1.0, scale=1.0, low=1e-3, high=100)
        space.setdefault("penalty", ["l2"])
        space.setdefault("solver", ["lbfgs", "saga"])
        space.setdefault("class_weight", [None, "balanced"])
    elif algo == "SVM":
        around_float("C", default=1.0, scale=1.0, low=1e-3, high=100)
        around_float("gamma", default= "scale" if isinstance(base.get("gamma", "scale"), str) else float(base.get("gamma", 0.1)), scale=1.0, low=1e-4, high=10.0)  # type: ignore
        space.setdefault("kernel", ["rbf", "linear", "poly"])
        space.setdefault("probability", [True])
        space.setdefault("class_weight", [None, "balanced"])
    elif algo == "XGBoostClassifier" and _XGB_AVAILABLE:
        around_int("n_estimators", default=300, low=50, high=1000, scale=1.0)
        around_float("learning_rate", default=0.1, scale=0.8, low=0.01, high=0.5)
        around_int("max_depth", default=6, low=3, high=12, scale=0.5)
        around_float("subsample", default=0.8, scale=0.5, low=0.5, high=1.0)
        around_float("colsample_bytree", default=0.8, scale=0.5, low=0.5, high=1.0)
        around_float("reg_alpha", default=0.0, scale=1.0, low=0.0, high=1.0)
        around_float("reg_lambda", default=1.0, scale=1.0, low=0.1, high=10.0)
else:
    if algo == "RandomForestRegressor":
        around_int("n_estimators", default=300, low=50, high=1000, scale=1.0)
        around_int("max_depth", default= None if base.get("max_depth") in (None, "None") else int(base.get("max_depth", 20)), low=3, high=64, scale=1.0)
        around_int("min_samples_split", default=2, low=2, high=20, scale=1.0)
        around_int("min_samples_leaf", default=1, low=1, high=10, scale=1.0)
        space.setdefault("max_features", ["sqrt", "log2", None])
        space.setdefault("bootstrap", [True, False])
    elif algo == "LinearRegression":
        # wenige Hyperparameter; keine Suche nötig
        space = {}
    elif algo == "SVR":
        around_float("C", default=1.0, scale=1.0, low=1e-3, high=100)
        around_float("epsilon", default=0.1, scale=1.0, low=1e-4, high=1.0)
        space.setdefault("kernel", ["rbf", "linear", "poly"])
        around_float("gamma", default= "scale" if isinstance(base.get("gamma", "scale"), str) else float(base.get("gamma", 0.1)), scale=1.0, low=1e-4, high=10.0)  # type: ignore
    elif algo == "XGBoostRegressor" and _XGB_AVAILABLE:
        around_int("n_estimators", default=300, low=50, high=1000, scale=1.0)
        around_float("learning_rate", default=0.1, scale=0.8, low=0.01, high=0.5)
        around_int("max_depth", default=6, low=3, high=12, scale=0.5)
        around_float("subsample", default=0.8, scale=0.5, low=0.5, high=1.0)
        around_float("colsample_bytree", default=0.8, scale=0.5, low=0.5, high=1.0)
        around_float("reg_alpha", default=0.0, scale=1.0, low=0.0, high=1.0)
        around_float("reg_lambda", default=1.0, scale=1.0, low=0.1, high=10.0)
return space

def select\_columns(df: pd.DataFrame, target\_col: str, feature\_names: Optional\[List\[str]]) -> Tuple\[pd.DataFrame, pd.Series]:
if feature\_names:
features = \[c for c in feature\_names if c in df.columns and c != target\_col]
missing = \[c for c in feature\_names if c not in df.columns]
if missing:
log(f"INFO: MissingFeaturesIgnored: {missing}")
X = df\[features].copy()
else:
X = df.drop(columns=\[target\_col]).copy()
y = df\[target\_col].copy()
return X, y

def build\_preprocessor(X: pd.DataFrame) -> Tuple\[ColumnTransformer, List\[str], List\[str]]:
num\_cols = X.select\_dtypes(include=\[np.number]).columns.tolist()
cat\_cols = X.select\_dtypes(include=\["object", "category", "bool"]).columns.tolist()
\# Falls jetzt noch datetime-ähnliche int/float Features fehlen, sind sie bereits numerisch
numeric\_pipeline = Pipeline(steps=\[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler(with\_mean=True, with\_std=True)),
])
categorical\_pipeline = Pipeline(steps=\[
("imputer", SimpleImputer(strategy="most\_frequent")),
("onehot", OneHotEncoder(handle\_unknown="ignore", sparse\_output=False)),
])
preprocessor = ColumnTransformer(
transformers=\[
("num", numeric\_pipeline, num\_cols),
("cat", categorical\_pipeline, cat\_cols),
],
remainder="drop",
verbose\_feature\_names\_out=False,
)
return preprocessor, num\_cols, cat\_cols

def encode\_target\_if\_needed(y: pd.Series) -> Tuple\[pd.Series, Optional\[LabelEncoder], Optional\[Dict\[Any, Any]]]:
if y.dtype.kind in ("O", "b") or y.dtype.name == "category":
le = LabelEncoder()
y\_enc = pd.Series(le.fit\_transform(y.astype(str)), index=y.index)
mapping = {cls: int(le.transform(\[cls])\[0]) for cls in le.classes\_}
return y\_enc, le, mapping
return y, None, None

def get\_cv(problem\_type: str, y: pd.Series, n\_splits: int = 5):
n = len(y)
n\_splits = min(n\_splits, max(2, min(10, n // 5)))
if problem\_type == "classification":
return StratifiedKFold(n\_splits=n\_splits, shuffle=True, random\_state=42)
return KFold(n\_splits=n\_splits, shuffle=True, random\_state=42)

def scoring\_for(problem\_type: str, n\_classes: Optional\[int]) -> str:
if problem\_type == "classification":
return "f1\_weighted" if (n\_classes and n\_classes > 2) else "roc\_auc\_ovr"
return "neg\_root\_mean\_squared\_error"

def evaluate\_classification(y\_true: np.ndarray, y\_pred: np.ndarray, proba: Optional\[np.ndarray]) -> None:
acc = accuracy\_score(y\_true, y\_pred)
prec = precision\_score(y\_true, y\_pred, average="weighted", zero\_division=0)
rec = recall\_score(y\_true, y\_pred, average="weighted", zero\_division=0)
f1 = f1\_score(y\_true, y\_pred, average="weighted", zero\_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
try:
n\_classes = len(np.unique(y\_true))
if proba is not None and n\_classes >= 2:
if n\_classes == 2 and proba.ndim == 1:
auc = roc\_auc\_score(y\_true, proba)
else:
auc = roc\_auc\_score(y\_true, proba, multi\_class="ovr")
print(f"ROC\_AUC: {auc:.4f}")
except Exception:
pass
cm = confusion\_matrix(y\_true, y\_pred)
log(f"ConfusionMatrix:\n{cm}")

def evaluate\_regression(y\_true: np.ndarray, y\_pred: np.ndarray) -> None:
r2 = r2\_score(y\_true, y\_pred)
rmse = math.sqrt(mean\_squared\_error(y\_true, y\_pred))
mae = mean\_absolute\_error(y\_true, y\_pred)
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
\# MAPE optional, robust gegen Division durch Null
try:
mape = np.mean(np.abs((y\_true - y\_pred) / np.clip(np.abs(y\_true), 1e-8, None)))
print(f"MAPE: {mape:.4f}")
except Exception:
pass

def main():
warnings.filterwarnings("ignore")
log(f"ProjectName: \${project.name}")
algorithm = "\${project.algorithm}"
model\_type\_declared = "\${project.modelType}"
target\_col = "\${project.targetVariable}"
features\_declared = "\${project.features.join(', ')}"
\# WICHTIG: Hyperparameter-JSON-String exakt wie gefordert
hyperparameters = "\${JSON.stringify(project.hyperparameters).replace(/"/g, '\\"')}"
source\_path = r"\${project.csvFilePath}"


# Parse Projekt-Features in Liste
feature_list = [f.strip() for f in features_declared.split(",")] if features_declared.strip() else []

# JSON -> Dict und numeric coercion
params_raw = json.loads(hyperparameters) if hyperparameters.strip() else {}
params = coerce_numeric_json_values(params_raw)

# Lade Daten
log(f"LoadingFile: {source_path}")
df = load_source(source_path)

# Grundbereinigung
df = basic_cleaning(df)

# Sicherstellen, dass Target existiert
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in data.")

# Feature-Auswahl
X, y = select_columns(df, target_col, feature_list if feature_list else None)

# Problem-Typ
problem_type = infer_problem_type(model_type_declared, y)

# Target-Encoding falls nötig
y_enc, target_encoder, target_mapping = encode_target_if_needed(y) if problem_type == "classification" else (y, None, None)

# Train/Test Split
test_size = 0.2
random_state = 42
if problem_type == "classification":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state
    )

# Preprocessor
preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

# Basismodell (mit gegebenen Hyperparametern)
base_estimator = build_estimator(algorithm, problem_type, params)

# Pipeline
pipe = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", base_estimator),
])

# Kurzes Fit mit Basis-Parametern für Baseline
log("INFO: Fitting base model with provided hyperparameters")
pipe.fit(X_train, y_train)
if problem_type == "classification":
    y_pred = pipe.predict(X_test)
    try:
        proba = pipe.predict_proba(X_test)
        proba_used = proba if proba.ndim > 1 else proba
    except Exception:
        try:
            scores = pipe.decision_function(X_test)
            # in binär: zu Wahrscheinlichkeiten mappen via logistic sigmoid als Approx
            if scores.ndim == 1:
                proba_used = 1 / (1 + np.exp(-scores))
            else:
                # Softmax
                e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                proba_used = e / np.sum(e, axis=1, keepdims=True)
        except Exception:
            proba_used = None
    log("BASELINE METRICS:")
    evaluate_classification(np.asarray(y_test), np.asarray(y_pred), proba_used)
else:
    y_pred = pipe.predict(X_test)
    log("BASELINE METRICS:")
    evaluate_regression(np.asarray(y_test), np.asarray(y_pred))

# Hyperparameter-Suche (um die angegebenen Parameter herum), falls sinnvoll
n_samples, n_features = X_train.shape
search_space = build_search_space(algorithm, problem_type, params, n_features, n_samples)
best_pipe = pipe
improved = False

if search_space:
    log("INFO: Starting hyperparameter tuning")
    cv = get_cv(problem_type, y_train)
    scoring = scoring_for(problem_type, int(len(np.unique(y_train))) if problem_type == "classification" else None)
    n_iter = min(30, max(10, int(0.2 * (len(pd.DataFrame(search_space).explode(list(search_space.keys())))))))
    try:
        rs = RandomizedSearchCV(
            estimator=Pipeline(steps=[("pre", preprocessor), ("model", build_estimator(algorithm, problem_type, params))]),
            param_distributions={f"model__{k}": v for k, v in search_space.items()},
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        rs.fit(X_train, y_train)
        best_pipe = rs.best_estimator_
        improved = True
        log(f"INFO: BestParams: {rs.best_params_}")
        log(f"INFO: BestCVScore: {rs.best_score_:.6f}")
    except Exception as e:
        log(f"WARNING: Hyperparameter tuning failed: {e}")

# Finale Evaluation mit ggf. getuntem Modell
if problem_type == "classification":
    y_pred_best = best_pipe.predict(X_test)
    try:
        proba = best_pipe.predict_proba(X_test)
        proba_used = proba if proba.ndim > 1 else proba
    except Exception:
        try:
            scores = best_pipe.decision_function(X_test)
            if scores.ndim == 1:
                proba_used = 1 / (1 + np.exp(-scores))
            else:
                e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                proba_used = e / np.sum(e, axis=1, keepdims=True)
        except Exception:
            proba_used = None
    log("FINAL METRICS:")
    evaluate_classification(np.asarray(y_test), np.asarray(y_pred_best), proba_used)
else:
    y_pred_best = best_pipe.predict(X_test)
    log("FINAL METRICS:")
    evaluate_regression(np.asarray(y_test), np.asarray(y_pred_best))

# Modell speichern
joblib.dump(best_pipe, "model.pkl")
log("INFO: SavedModel: model.pkl")

# Target-Encoder speichern (falls benutzt)
if target_encoder is not None:
    enc_path = f"target_encoder_${project.id}.pkl"
    joblib.dump(target_encoder, enc_path)
    log(f"INFO: SavedTargetEncoder: {enc_path}")
    if target_mapping:
        log(f"INFO: TargetMapping: {target_mapping}")


if **name** == "**main**":
main()
```	


export const codeReviewSystem = (code) => `Du bist ein strenger Python-Reviewer.

Hier ist der Code, der überprüft werden soll:
${code}

Prüfe den gegebenen Python-Code auf:
- Syntaxfehler, fehlende Imports
- Pfadprobleme (Datei nicht gefunden)
- Edge Cases (leere Spalten, eine Klasse)
Zusätzliche harte Regeln:
- Verwende KEINEN LabelEncoder in Feature-Transformern/ColumnTransformer/Pipelines (LabelEncoder ist ausschließlich für das Target y erlaubt)
- Für kategoriale Features IMMER OneHotEncoder(handle_unknown='ignore', sparse_output=False) verwenden
- Für numerische Features IMMER SimpleImputer(strategy='median') und StandardScaler verwenden
- Trenne X (Features) und y (Target) korrekt; trainiere die Pipeline mit X und y, nicht mit einem gesamten DataFrame 'data'
- Verwende niemals pipeline.fit_transform(data), sondern trenne in Training/Test und nutze pipeline.fit(X_train, y_train) und pipeline.transform(X_test)
Gib bei Problemen ein minimal-invasives, aber lauffähiges, KORRIGIERTES Skript zurück.
Gib nur ausschließlich Python-Code zurück. Alle anderen Texte und Erklärungen sind ausdrücklich unerwünscht und müssen auskommentiert werden mit #.`;

export const domainHPSystem = (dataAnalysis) => `Du bist ein Fachdomänen-Experte.

# ${dataAnalysis.llm_summary}

Basierend auf Datenanalyse (Zielvariable, Feature-Typen, Balance) schlage präzisere Hyperparameter vor.
Gib ein JSON mit Schlüssel/Wert zurück, Werte als numerische Typen wo sinnvoll.`;
