import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Optional boosters
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Keras/TensorFlow optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -----------------------
# Config
# -----------------------
class Config:
    DATA_PATH = "fault_data.csv"
    SHUFFLED_DATA_PATH = "fault_data_shuffled.csv"
    MODELS = "saved_models"
    RESULTS = "results"
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    N_JOBS = -1

    # DL
    DL_EPOCHS = 50
    DL_BATCH_SIZE = 1024
    DL_EARLYSTOP_PATIENCE = 8

    # Quick experiment limit (set None to use full dataset)
    MAX_SAMPLES = 200000  # CHANGE as needed

    # Use SMOTE oversampling for minority classes? (only for classical & DL training)
    USE_SMOTE = False  # toggle; requires imblearn

    # Shuffle and persist dataset before training (recommended for ordered datasets)
    SHUFFLE_BEFORE_TRAIN = True

Path(Config.MODELS).mkdir(exist_ok=True)
Path(Config.RESULTS).mkdir(exist_ok=True)

np.random.seed(Config.RANDOM_STATE)
sns.set()

# -----------------------
# Helpers
# -----------------------
def save_figure(fig, name):
    out = f"{Config.RESULTS}/{name}"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", out)

def plot_confusion_matrix_matrix(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    save_figure(fig, fname)

def dump_model(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved model to", path)

def shuffle_and_save_dataset(input_path, output_path, random_state=42):
    print("Shuffling dataset before training...")
    df = pd.read_csv(input_path)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled.to_csv(output_path, index=False)
    print(f"Saved shuffled dataset: {output_path} (rows={len(df_shuffled)})")
    return output_path

# -----------------------
# Derived/phase label builder
# -----------------------
def build_phase_label(row):
    parts = []
    if int(row['A']) == 1:
        parts.append('A')
    if int(row['B']) == 1:
        parts.append('B')
    if int(row['C']) == 1:
        parts.append('C')
    if int(row['G']) == 1:
        parts.append('G')
    if len(parts) == 0:
        return "NONE"
    return "".join(parts)

# -----------------------
# Feature engineering (RMS-safe; no angles)
# -----------------------
def prepare_features(df):
    df = df.copy()
    base_feats = ['Ia','Ib','Ic','Va','Vb','Vc']
    for c in base_feats:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    # ------------------ BASIC POWER ------------------
    df['P1'] = df['Ia'] * df['Va']
    df['P2'] = df['Ib'] * df['Vb']
    df['P3'] = df['Ic'] * df['Vc']
    df['P_total'] = df['P1'] + df['P2'] + df['P3']

    # ------------------ DIFFERENCES ------------------
    df['dV_ab'] = df['Va'] - df['Vb']
    df['dV_bc'] = df['Vb'] - df['Vc']
    df['dV_ca'] = df['Vc'] - df['Va']
    df['dI_ab'] = df['Ia'] - df['Ib']
    df['dI_bc'] = df['Ib'] - df['Ic']
    df['dI_ca'] = df['Ic'] - df['Ia']

    eps = 1e-8
    df['V_ratio_ab'] = (df['Va']+eps)/(df['Vb']+eps)
    df['I_ratio_ab'] = (df['Ia']+eps)/(df['Ib']+eps)

    df['V_max_min_diff'] = df[['Va','Vb','Vc']].max(axis=1) - df[['Va','Vb','Vc']].min(axis=1)
    df['I_max_min_diff'] = df[['Ia','Ib','Ic']].max(axis=1) - df[['Ia','Ib','Ic']].min(axis=1)

    df['V_mean'] = df[['Va','Vb','Vc']].mean(axis=1)
    df['V_std']  = df[['Va','Vb','Vc']].std(axis=1)
    df['I_mean'] = df[['Ia','Ib','Ic']].mean(axis=1)
    df['I_std']  = df[['Ia','Ib','Ic']].std(axis=1)

    # ------------------  ZERO-SEQUENCE FEATURES ------------------
    df['I0'] = (df['Ia'] + df['Ib'] + df['Ic']) / 3
    df['V0'] = (df['Va'] + df['Vb'] + df['Vc']) / 3

    df['I0_abs'] = np.abs(df['I0'])
    df['V0_abs'] = np.abs(df['V0'])

    df['I0_ratio'] = df['I0_abs'] / (df['I_mean'] + eps)
    df['V0_ratio'] = df['V0_abs'] / (df['V_mean'] + eps)

    df['ground_imbalance'] = np.abs(df['Ia'] + df['Ib'] + df['Ic'])

    # ------------------ NEGATIVE SEQUENCE APPROX ------------------
    df['neg_seq_approx'] = (
        np.abs(df['Ia'] - df['Ib']) +
        np.abs(df['Ib'] - df['Ic']) +
        np.abs(df['Ic'] - df['Ia'])
    )

    # ------------------ GROUND POWER ------------------
    df['Pg'] = df['I0'] * df['V0']
    df['log_Pg'] = np.log(np.abs(df['Pg']) + 1e-6)

    # ------------------ INTERACTIONS ------------------
    df['Va_Ia'] = df['Va'] * df['Ia']
    df['Vb_Ib'] = df['Vb'] * df['Ib']
    df['Vc_Ic'] = df['Vc'] * df['Ic']

    for c in ['P1','P2','P3','P_total','V_max_min_diff','I_max_min_diff']:
        df[f'log_{c}'] = np.log(np.abs(df[c]) + 1e-6)

    final_feats = base_feats + [
        'P1','P2','P3','P_total',
        'dV_ab','dV_bc','dV_ca','dI_ab','dI_bc','dI_ca',
        'V_ratio_ab','I_ratio_ab',
        'V_max_min_diff','I_max_min_diff',
        'V_mean','V_std','I_mean','I_std',

        # new critical features
        'I0','V0','I0_abs','V0_abs',
        'I0_ratio','V0_ratio',
        'ground_imbalance',
        'neg_seq_approx',
        'Pg','log_Pg',

        'Va_Ia','Vb_Ib','Vc_Ic',
        'log_P1','log_P2','log_P3','log_P_total'
    ]

    final_feats = [f for f in final_feats if f in df.columns]
    return df[final_feats], final_feats


# -----------------------
# Load & preprocess dataset
# -----------------------
def load_dataset(path, max_samples=None):
    print("Loading dataset:", path)
    df = pd.read_csv(path)
    print("Original shape:", df.shape)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=Config.RANDOM_STATE).reset_index(drop=True)
        print("Using sample:", df.shape)

    # check for required columns
    req = ['fault_type','location','A','B','C','G','Ia','Ib','Ic','Va','Vb','Vc']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # normalize labels
    df['fault_type'] = df['fault_type'].astype(str).str.upper().str.strip()
    df['location'] = df['location'].astype(str).str.strip().str.lower()

    # derive phase label
    df['phase_label'] = df.apply(build_phase_label, axis=1)

    return df

# -----------------------
# Encoders / maps
# -----------------------
FAULT_TYPE_ORDER = ['NOFAULT','LG','LL','LLG','LLL','LLLG']   # known set
LOCATION_ORDER = ['near_source','mid_line','near_load']      # assumed from earlier output

fault_le = LabelEncoder().fit(FAULT_TYPE_ORDER)
loc_le = LabelEncoder().fit(LOCATION_ORDER)
phase_le = LabelEncoder()  # will fit to dataset

# -----------------------
# Classical model builders (with tuned params)
# -----------------------
def build_classic_models():
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=Config.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=Config.N_JOBS, random_state=Config.RANDOM_STATE)
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=Config.N_JOBS,
            tree_method='hist',
            random_state=Config.RANDOM_STATE
        )
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=128,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=Config.N_JOBS,
            random_state=Config.RANDOM_STATE
        )
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = cb.CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=8,
            verbose=False,
            random_seed=Config.RANDOM_STATE
        )
    return models

# -----------------------
# DL model (larger + residual like)
# -----------------------
def build_dl_model(input_dim, n_fault, n_loc, n_phase):
    inp = Input(shape=(input_dim,), name='input')
    x = Dense(512, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x2 = Dense(256, activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)

    # residual concat
    x = Concatenate()([x, x2])
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # heads
    hf = Dense(128, activation='relu')(x)
    out_fault = Dense(n_fault, activation='softmax', name='fault_type')(hf)

    hl = Dense(128, activation='relu')(x)
    out_loc = Dense(n_loc, activation='softmax', name='location')(hl)

    hp = Dense(128, activation='relu')(x)
    out_phase = Dense(n_phase, activation='softmax', name='phase')(hp)

    model = Model(inputs=inp, outputs=[out_fault, out_loc, out_phase])
    model.compile(
        optimizer=Adam(1e-3),
        loss={'fault_type':'sparse_categorical_crossentropy','location':'sparse_categorical_crossentropy','phase':'sparse_categorical_crossentropy'},
        metrics={'fault_type':'accuracy','location':'accuracy','phase':'accuracy'}
    )
    return model

# -----------------------
# Training helpers
# -----------------------
def compute_sample_weights(y, classes):
    # compute class weights and convert to vector per-sample
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    cw = {cls: w for cls, w in zip(np.unique(y), weights)}
    sample_w = np.array([cw[int(x)] for x in y])
    # normalize
    sample_w = sample_w / np.mean(sample_w)
    return sample_w, cw

def train_classical_task(X_train, y_train, X_test, y_test, task_name, class_names):
    results = {}
    models = build_classic_models()
    for name, m in models.items():
        print(f"\nTraining classical {task_name} - {name}")
        # Fit with default (we can pass sample weights if needed)
        try:
            m.fit(X_train, y_train)
        except Exception as e:
            print("Model training failed:", e)
            continue
        pred = m.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, pred, average='weighted')
        print(f"{task_name} {name}: ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}, F1={f1:.4f}")
        results[name] = {'model': m, 'pred': pred, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    return results

# -----------------------
# Main
# -----------------------
def main():
    train_data_path = Config.DATA_PATH
    if Config.SHUFFLE_BEFORE_TRAIN:
        train_data_path = shuffle_and_save_dataset(
            Config.DATA_PATH,
            Config.SHUFFLED_DATA_PATH,
            random_state=Config.RANDOM_STATE
        )

    # 1) load
    df = load_dataset(train_data_path, max_samples=Config.MAX_SAMPLES)
    print("Sample of labels:")
    print(df[['fault_type','location','phase_label']].head())

    # 2) fit phase label encoder
    phase_le.fit(df['phase_label'].unique())
    print("Phase classes:", list(phase_le.classes_))

    # 3) encode targets
    # ensure fault types are subset of FAULT_TYPE_ORDER; otherwise extend mapping
    uniq_ft = set(df['fault_type'].unique())
    missing_ft = uniq_ft - set(FAULT_TYPE_ORDER)
    if missing_ft:
        print("Warning: Unknown fault types found; they will be added to encoder:", missing_ft)
        # extend
        combined = list(FAULT_TYPE_ORDER) + sorted(list(missing_ft))
        fault_le.fit(combined)

    y_fault = fault_le.transform(df['fault_type'])
    # location mapping - try to map strings to expected order; if unseen values, warn
    uniq_loc = set(df['location'].unique())
    if not uniq_loc.issubset(set(LOCATION_ORDER)):
        print("Warning: location values outside expected LOCATION_ORDER:", uniq_loc)
        # fit loc_le to dataset's unique locations to avoid crash
        loc_le.fit(sorted(list(uniq_loc)))
    y_loc = loc_le.transform(df['location'])
    y_phase = phase_le.transform(df['phase_label'])

    # 4) features
    Xdf, feat_names = prepare_features(df)
    print("Using features:", feat_names)

    # 5) scale with RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X = scaler.fit_transform(Xdf.values)

    # 6) train/test/val split (stratify on fault type for balanced faults)
    Xtemp, Xtest, yft_temp, yft_test, yloc_temp, yloc_test, yph_temp, yph_test = train_test_split(
        X, y_fault, y_loc, y_phase, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y_fault
    )
    val_ratio = Config.VAL_SIZE / (1 - Config.TEST_SIZE)
    Xtrain, Xval, yft_train, yft_val, yloc_train, yloc_val, yph_train, yph_val = train_test_split(
        Xtemp, yft_temp, yloc_temp, yph_temp, test_size=val_ratio,
        random_state=Config.RANDOM_STATE, stratify=yft_temp
    )
    print("Shapes: train", Xtrain.shape, "val", Xval.shape, "test", Xtest.shape)

    # Optionally apply SMOTE to training set (only if enabled & available)
    if Config.USE_SMOTE:
        if not IMBLEARN_AVAILABLE:
            print("SMOTE requested but imblearn not available — skipping SMOTE. Install imblearn to enable.")
        else:
            print("Applying SMOTE to training data (phase labels)...")
            # We'll apply SMOTE on the phase label task (which often is the most imbalanced).
            # Note: SMOTE requires moderate dataset size; use subset if too large.
            sm = SMOTE(random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS)
            Xtrain, yph_train = sm.fit_resample(Xtrain, yph_train)
            # We need to align other targets (fault_type, location) by mapping original indices:
            # This is tricky because SMOTE creates synthetic samples — we can't reliably resample other labels.
            # Simpler approach: perform SMOTE separately per-task when training classical models (not used here).
            print("After SMOTE: Xtrain shape:", Xtrain.shape)
            # Note: for multi-task DL, SMOTE may not be ideal. Alternative: use sample weights.

    # 7) Classical models (per-task)
    final_rows = []

    # Fault type classical
    print("\n--- TRAINING CLASSICAL MODELS FOR FAULT TYPE ---")
    ft_results = train_classical_task(Xtrain, yft_train, Xtest, yft_test, "fault_type", list(fault_le.classes_))
    for name, r in ft_results.items():
        final_rows.append([f"FT_{name}", "FaultType", r['acc'], r['prec'], r['rec'], r['f1'], "Classical"])
        plot_confusion_matrix_matrix(yft_test, r['pred'], list(fault_le.classes_), f"FT_{name}", f"cm_FT_{name}.png")
        dump_model(r['model'], f"{Config.MODELS}/FT_{name}.pkl")

    # Location classical
    print("\n--- TRAINING CLASSICAL MODELS FOR LOCATION ---")
    fl_results = train_classical_task(Xtrain, yloc_train, Xtest, yloc_test, "location", list(loc_le.classes_))
    for name, r in fl_results.items():
        final_rows.append([f"FL_{name}", "Location", r['acc'], r['prec'], r['rec'], r['f1'], "Classical"])
        plot_confusion_matrix_matrix(yloc_test, r['pred'], list(loc_le.classes_), f"FL_{name}", f"cm_FL_{name}.png")
        dump_model(r['model'], f"{Config.MODELS}/FL_{name}.pkl")

    # Phase classical
    print("\n--- TRAINING CLASSICAL MODELS FOR PHASE ---")
    ph_results = train_classical_task(Xtrain, yph_train, Xtest, yph_test, "phase", list(phase_le.classes_))
    for name, r in ph_results.items():
        final_rows.append([f"PH_{name}", "Phase", r['acc'], r['prec'], r['rec'], r['f1'], "Classical"])
        plot_confusion_matrix_matrix(yph_test, r['pred'], list(phase_le.classes_), f"PH_{name}", f"cm_PH_{name}.png")
        dump_model(r['model'], f"{Config.MODELS}/PH_{name}.pkl")

    # 8) Deep multi-task model
    if TF_AVAILABLE:
        print("\n--- BUILDING & TRAINING DEEP MULTI-TASK MODEL ---")
        dl_model = build_dl_model(input_dim=Xtrain.shape[1],
                                 n_fault=len(fault_le.classes_),
                                 n_loc=len(loc_le.classes_),
                                 n_phase=len(phase_le.classes_))
        dl_model.summary()

        # compute per-head sample weights to mitigate imbalance
        sw_fault, cw_fault = compute_sample_weights(yft_train, np.unique(yft_train))
        sw_loc, cw_loc = compute_sample_weights(yloc_train, np.unique(yloc_train))
        sw_phase, cw_phase = compute_sample_weights(yph_train, np.unique(yph_train))

        # Keep output order aligned with model outputs: [fault_type, location, phase]
        y_train_list = [yft_train, yloc_train, yph_train]
        y_val_list = [yft_val, yloc_val, yph_val]
        sample_weight_list = [sw_fault, sw_loc, sw_phase]

        ckpt = f"{Config.MODELS}/dl_multitask_best.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=Config.DL_EARLYSTOP_PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        history = dl_model.fit(
            Xtrain, y_train_list,
            validation_data=(Xval, y_val_list),
            epochs=Config.DL_EPOCHS,
            batch_size=Config.DL_BATCH_SIZE,
            callbacks=callbacks,
            sample_weight=sample_weight_list,
            verbose=2
        )

        # plot training curves
        fig, axes = plt.subplots(2,1,figsize=(10,9))
        axes[0].plot(history.history['loss'], label='train_loss')
        axes[0].plot(history.history['val_loss'], label='val_loss')
        axes[0].legend(); axes[0].set_title("Total loss")

        # plot each head accuracy if present
        for metric_key in ['fault_type_accuracy','val_fault_type_accuracy','location_accuracy','val_location_accuracy','phase_accuracy','val_phase_accuracy']:
            if metric_key in history.history:
                axes[1].plot(history.history[metric_key], label=metric_key)
        axes[1].legend(); axes[1].set_title("Per-head accuracies")
        save_figure(fig, "dl_training_curves.png")

        # predict & evaluate
        preds = dl_model.predict(Xtest, batch_size=Config.DL_BATCH_SIZE, verbose=0)
        pred_fault = np.argmax(preds[0], axis=1)
        pred_loc = np.argmax(preds[1], axis=1)
        pred_phase = np.argmax(preds[2], axis=1)

        f1_ft = f1_score(yft_test, pred_fault, average='weighted')
        acc_ft = accuracy_score(yft_test, pred_fault)
        prec_ft = precision_score(yft_test, pred_fault, average='weighted', zero_division=0)
        rec_ft = recall_score(yft_test, pred_fault, average='weighted', zero_division=0)
        print("DL FaultType ACC/F1:", acc_ft, f1_ft)
        final_rows.append(["FT_DL", "FaultType", acc_ft, prec_ft, rec_ft, f1_ft, "DeepLearning"])
        plot_confusion_matrix_matrix(yft_test, pred_fault, list(fault_le.classes_), "DL FaultType", "cm_DL_FT.png")

        f1_loc = f1_score(yloc_test, pred_loc, average='weighted')
        acc_loc = accuracy_score(yloc_test, pred_loc)
        prec_loc = precision_score(yloc_test, pred_loc, average='weighted', zero_division=0)
        rec_loc = recall_score(yloc_test, pred_loc, average='weighted', zero_division=0)
        print("DL Location ACC/F1:", acc_loc, f1_loc)
        final_rows.append(["FL_DL", "Location", acc_loc, prec_loc, rec_loc, f1_loc, "DeepLearning"])
        plot_confusion_matrix_matrix(yloc_test, pred_loc, list(loc_le.classes_), "DL Loc", "cm_DL_FL.png")

        f1_ph = f1_score(yph_test, pred_phase, average='weighted')
        acc_ph = accuracy_score(yph_test, pred_phase)
        prec_ph = precision_score(yph_test, pred_phase, average='weighted', zero_division=0)
        rec_ph = recall_score(yph_test, pred_phase, average='weighted', zero_division=0)
        print("DL Phase ACC/F1:", acc_ph, f1_ph)
        final_rows.append(["PH_DL", "Phase", acc_ph, prec_ph, rec_ph, f1_ph, "DeepLearning"])
        plot_confusion_matrix_matrix(yph_test, pred_phase, list(phase_le.classes_), "DL Phase", "cm_DL_PH.png")

        # save model
        dl_model.save(ckpt)
        print("Saved DL model to", ckpt)
    else:
        print("TensorFlow not available — skipping DL model training.")

    # 9) Final ranking
    final_df = pd.DataFrame(
        final_rows,
        columns=["Model", "Task", "Accuracy", "Precision", "Recall", "F1", "Type"]
    )
    if not final_df.empty:
        final_df_sorted = final_df.sort_values(by="F1", ascending=False)
        final_df_sorted.to_csv(f"{Config.RESULTS}/model_ranking_improved.csv", index=False)
        print("\nFinal ranking:")
        print(final_df_sorted.to_string(index=False))
    else:
        print("No results to show in final ranking.")

    print("\nAll done.")

if __name__ == "__main__":
    main()
