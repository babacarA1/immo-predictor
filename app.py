"""
Immo Predictor — Application Streamlit
Lancer :python -m streamlit run app.py
Prérequis : pip install streamlit scikit-learn pandas numpy matplotlib seaborn
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)


# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION PAGE
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Immo Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; line-height: 1.2;
    }
    .subtitle { color: #6b7280; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border-left: 4px solid #1a73e8; border-radius: 12px;
        padding: 1rem 1.2rem; margin: 0.4rem 0;
        box-shadow: 0 2px 8px rgba(26,115,232,0.08);
    }
    .kpi-label { font-size: 0.78rem; color: #6b7280; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #1a73e8;
                  line-height: 1.1; }
    .kpi-sub   { font-size: 0.75rem; color: #9ca3af; margin-top: 2px; }
    .result-price {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #2e7d32; border-radius: 14px;
        padding: 1.8rem; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46,125,50,0.12);
    }
    .result-price-val { font-size: 3rem; font-weight: 900; color: #1b5e20; }
    .result-type {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #0d47a1; border-radius: 14px;
        padding: 1.8rem; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(13,71,161,0.12);
    }
    .result-type-val { font-size: 2rem; font-weight: 800; color: #0d47a1; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1f2937;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .badge-green {
        background: #dcfce7; color: #166534; border-radius: 20px;
        padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    }
    .badge-blue {
        background: #dbeafe; color: #1e40af; border-radius: 20px;
        padding: 2px 10px; font-size: 0.8rem; font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem; font-weight: 600; padding: 0.6rem 1.2rem;
    }
    div[data-testid="stExpander"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────
REG_FEATURES = [
    'GrLivArea', 'TotalBsmtSF', 'LotArea', 'BedroomAbvGr', 'FullBath',
    'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'Neighborhood', 'GarageCars', 'GarageArea', 'PoolArea', 'Fireplaces'
]
CLF_FEATURES = ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'YearBuilt',
                 'GarageCars', 'Neighborhood', 'HouseStyle']

BLDG_LABELS = {
    '1Fam':   '🏡 Maison Individuelle',
    '2fmCon': '🏘️ Maison 2 Logements',
    'Duplex': '🏢 Duplex',
    'TwnhsE': '🏙️ Townhouse (bout de rangée)',
    'Twnhs':  '🏙️ Townhouse (intérieur)'
}

# ─────────────────────────────────────────────────────────────────────
#  CHARGEMENT & ENTRAÎNEMENT (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Entraînement des modèles en cours…")
def load_and_train(csv_path):
    df = pd.read_csv(csv_path)

    # ════════ RÉGRESSION ════════
    df_r = df[REG_FEATURES + ['SalePrice']].copy()

    le_neigh = LabelEncoder()
    df_r['Neighborhood'] = le_neigh.fit_transform(df_r['Neighborhood'].astype(str))

    imp_r = SimpleImputer(strategy='median')
    X_r = pd.DataFrame(imp_r.fit_transform(df_r[REG_FEATURES]), columns=REG_FEATURES)
    y_r = df_r['SalePrice'].reset_index(drop=True)

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    dt_r = DecisionTreeRegressor(max_depth=10, min_samples_split=5,
                                  min_samples_leaf=3, random_state=42)
    dt_r.fit(Xtr_r, ytr_r); dt_pred = dt_r.predict(Xte_r)

    rf_r = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                                  random_state=42, n_jobs=-1)
    rf_r.fit(Xtr_r, ytr_r); rf_pred = rf_r.predict(Xte_r)

    reg_res = {
        'Decision Tree': {
            'MAE': mean_absolute_error(yte_r, dt_pred),
            'RMSE': np.sqrt(mean_squared_error(yte_r, dt_pred)),
            'R2': r2_score(yte_r, dt_pred), 'preds': dt_pred
        },
        'Random Forest': {
            'MAE': mean_absolute_error(yte_r, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(yte_r, rf_pred)),
            'R2': r2_score(yte_r, rf_pred), 'preds': rf_pred
        }
    }

    # ════════ CLASSIFICATION ════════
    df_c = df[CLF_FEATURES + ['BldgType']].copy()

    le_nc = LabelEncoder(); le_hs = LabelEncoder(); le_bt = LabelEncoder()
    df_c['Neighborhood'] = le_nc.fit_transform(df_c['Neighborhood'].astype(str))
    df_c['HouseStyle']   = le_hs.fit_transform(df_c['HouseStyle'].astype(str))
    y_c = le_bt.fit_transform(df_c['BldgType'])
    classes = list(le_bt.classes_)

    imp_c = SimpleImputer(strategy='median')
    X_c = pd.DataFrame(imp_c.fit_transform(df_c[CLF_FEATURES]), columns=CLF_FEATURES)
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(X_c)

    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        Xsc, y_c, test_size=0.2, random_state=42, stratify=y_c)

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(Xtr_c, ytr_c); svm_pred = svm.predict(Xte_c)

    rf_c = RandomForestClassifier(n_estimators=200, random_state=42,
                                   n_jobs=-1, class_weight='balanced')
    rf_c.fit(Xtr_c, ytr_c); rf_c_pred = rf_c.predict(Xte_c)

    clf_res = {
        'SVM (RBF)': {
            'Accuracy': accuracy_score(yte_c, svm_pred),
            'F1': f1_score(yte_c, svm_pred, average='weighted'),
            'CM': confusion_matrix(yte_c, svm_pred),
            'report': classification_report(yte_c, svm_pred, target_names=classes)
        },
        'Random Forest': {
            'Accuracy': accuracy_score(yte_c, rf_c_pred),
            'F1': f1_score(yte_c, rf_c_pred, average='weighted'),
            'CM': confusion_matrix(yte_c, rf_c_pred),
            'report': classification_report(yte_c, rf_c_pred, target_names=classes)
        }
    }

    return {
        'df': df,
        # Régression
        'rf_r': rf_r, 'dt_r': dt_r, 'imp_r': imp_r, 'le_neigh': le_neigh,
        'X_r': X_r, 'yte_r': yte_r, 'reg_res': reg_res,
        'fi_r': pd.Series(rf_r.feature_importances_, index=REG_FEATURES).sort_values(ascending=False),
        # Classification
        'rf_c': rf_c, 'svm': svm, 'imp_c': imp_c, 'scaler': scaler,
        'le_nc': le_nc, 'le_hs': le_hs, 'le_bt': le_bt,
        'clf_res': clf_res, 'classes': classes,
        'fi_c': pd.Series(rf_c.feature_importances_, index=CLF_FEATURES).sort_values(ascending=False),
    }

# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏠 Immo Predictor")
    st.markdown("---")
    st.markdown("**📂 Dataset**")
    uploaded = st.file_uploader("Importer train.csv", type="csv",
                                  help="Télécharger depuis Kaggle : lespin/house-prices-dataset")
    st.markdown("---")
    st.markdown("**🧭 Navigation**")
    page = st.radio("Navigation", [
        "🏠  Accueil",
        "📊  Analyse EDA",
        "📐  Régression",
        "🏷️   Classification",
        "🔮  Prédicteur"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <small>
    📖 <b>Dataset</b> : House Prices (Kaggle)<br>
    🧠 <b>Modèles</b> : Decision Tree, Random Forest, SVM<br>
    🐍 <b>Stack</b> : Python · Sklearn · Streamlit
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────────────────────────────
if uploaded:
    import tempfile, os, io
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp.write(uploaded.read()); tmp.close()
    try:
        M = load_and_train(tmp.name)
    except Exception as e:
        st.error(f"❌ Erreur : {e}"); st.stop()
else:
    try:
        M = load_and_train('train.csv')
    except FileNotFoundError:
        st.error("⚠️ Veuillez importer **train.csv** via la barre latérale.")
        st.info("Télécharger sur : https://www.kaggle.com/datasets/lespin/house-prices-dataset")
        st.stop()

df = M['df']

# ─────────────────────────────────────────────────────────────────────
#  PAGE : ACCUEIL
# ─────────────────────────────────────────────────────────────────────
if "Accueil" in page:
    st.markdown('<div class="main-title">🏠 Immo Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Plateforme de Valorisation et Diagnostic Intelligent par Machine Learning</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    best_r = M['reg_res']['Random Forest']
    best_c = M['clf_res']['Random Forest']
    kpis = [
        (c1, "Observations", f"{len(df):,}", "maisons analysées"),
        (c2, "Variables", f"{df.shape[1]}", "features disponibles"),
        (c3, "R² Régression", f"{best_r['R2']:.3f}", f"RMSE ${best_r['RMSE']:,.0f}"),
        (c4, "Accuracy Classif.", f"{best_c['Accuracy']:.1%}", f"F1 {best_c['F1']:.3f}")
    ]
    for col, label, val, sub in kpis:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        ### 📐 Partie 1 — Régression
        **Prédiction du prix de vente** (`SalePrice`)

        | Modèle | Métriques |
        |---|---|
        | Decision Tree | MAE, RMSE, R² |
        | **Random Forest** ✅ | MAE, RMSE, R² |

        → 15 variables · Split 80/20 · Validation croisée 5-folds
        """)
    with cb:
        st.markdown("""
        ### 🏷️ Partie 2 — Classification
        **Classification du type de bâtiment** (`BldgType`)

        | Modèle | Métriques |
        |---|---|
        | SVM (kernel RBF) | Accuracy, F1, Confusion |
        | **Random Forest** ✅ | Accuracy, F1, Confusion |

        → 7 variables · Split stratifié · 5 classes
        """)

    st.markdown("---")
    st.markdown("### 🚀 Comment utiliser cette application")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**① EDA**\nExplorer les données, distributions et corrélations")
    col2.success("**② Régression**\nComparer DT vs RF sur la prédiction du prix")
    col3.warning("**③ Classification**\nComparer SVM vs RF sur le type de bâtiment")
    col4.error("**④ Prédicteur**\nSaisir un bien → obtenir prix + type estimés")

# ─────────────────────────────────────────────────────────────────────
#  PAGE : EDA
# ─────────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("## 📊 Analyse Exploratoire des Données")
    tab1, tab2, tab3 = st.tabs(["💰 SalePrice", "🏘️ Neighborhood", "🏷️ Classification"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df['SalePrice'], bins=50, color='#1a73e8', edgecolor='white', alpha=0.85)
            ax.set_title('Distribution de SalePrice', fontweight='bold')
            ax.set_xlabel('Prix ($)'); ax.set_ylabel('Fréquence')
            st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(np.log1p(df['SalePrice']), bins=50, color='#ff6d00', edgecolor='white', alpha=0.85)
            ax.set_title('Distribution de log(SalePrice + 1)', fontweight='bold')
            ax.set_xlabel('log(Prix)')
            st.pyplot(fig); plt.close()

        reg_num = ['GrLivArea','TotalBsmtSF','LotArea','OverallQual','OverallCond',
                   'GarageArea','GarageCars','FullBath','Fireplaces','SalePrice']
        fig, ax = plt.subplots(figsize=(11, 7))
        mask = np.triu(np.ones((len(reg_num), len(reg_num)), dtype=bool))
        sns.heatmap(df[reg_num].corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', ax=ax, linewidths=0.4, square=True)
        ax.set_title('Matrice de Corrélation — Variables Numériques', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Scatter plots — Top variables vs SalePrice**")
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        vars_scat = [('GrLivArea','#1a73e8'), ('OverallQual','#9c27b0'),
                     ('GarageArea','#ff6d00'), ('YearBuilt','#2e7d32')]
        for ax, (var, c) in zip(axes, vars_scat):
            ax.scatter(df[var], df['SalePrice'], alpha=0.3, s=12, color=c)
            ax.set_title(f'{var} vs SalePrice', fontweight='bold', fontsize=10)
            ax.set_xlabel(var); ax.set_ylabel('SalePrice')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        neigh_med = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
        neigh_med.plot(kind='bar', ax=axes[0], color='#1a73e8', edgecolor='white', alpha=0.85)
        axes[0].set_title('Prix Médian par Quartier', fontweight='bold')
        axes[0].set_ylabel('Prix Médian ($)'); axes[0].tick_params(axis='x', rotation=45)

        neigh_cnt = df['Neighborhood'].value_counts()
        neigh_cnt.plot(kind='bar', ax=axes[1], color='#ff6d00', edgecolor='white', alpha=0.85)
        axes[1].set_title('Nombre de Biens par Quartier', fontweight='bold')
        axes[1].set_ylabel('Nombre'); axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_bt = ['#1a73e8','#2e7d32','#ff6d00','#9c27b0','#e74c3c']
            df['BldgType'].value_counts().plot(kind='bar', ax=ax, color=colors_bt, edgecolor='white')
            ax.set_title('Distribution de BldgType', fontweight='bold')
            ax.set_ylabel('Nombre'); ax.tick_params(axis='x', rotation=30)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.boxplot(column='GrLivArea', by='BldgType', ax=ax)
            ax.set_title('Surface Habitable par Type', fontweight='bold')
            plt.suptitle(''); ax.set_ylabel('GrLivArea (sq ft)')
            ax.tick_params(axis='x', rotation=20)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.countplot(data=df, x='HouseStyle', hue='BldgType', palette='Set2', ax=ax)
        ax.set_title('Style de Maison par Type de Bâtiment', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────
#  PAGE : RÉGRESSION
# ─────────────────────────────────────────────────────────────────────
elif "Régression" in page:
    st.markdown("## 📐 Résultats — Régression : Prédiction de SalePrice")

    res = M['reg_res']; yte = M['yte_r']

    # KPIs
    best = res['Random Forest']
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">MAE — Random Forest</div>
        <div class="kpi-value">${best['MAE']:,.0f}</div>
        <div class="kpi-sub">Erreur absolue moyenne</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">RMSE — Random Forest</div>
        <div class="kpi-value">${best['RMSE']:,.0f}</div>
        <div class="kpi-sub">Root Mean Squared Error</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">R² — Random Forest</div>
        <div class="kpi-value">{best['R2']:.4f}</div>
        <div class="kpi-sub">Coefficient de détermination</div>
    </div>""", unsafe_allow_html=True)

    # Tableau
    st.markdown('<div class="section-title">Tableau Comparatif</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        'Modèle': list(res.keys()),
        'MAE ($)': [f"{v['MAE']:,.0f}" for v in res.values()],
        'RMSE ($)': [f"{v['RMSE']:,.0f}" for v in res.values()],
        'R²': [f"{v['R2']:.4f}" for v in res.values()]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Scatter prédictions vs réelles
    st.markdown('<div class="section-title">Prédictions vs Valeurs Réelles</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#1a73e8', '#2e7d32']
    for ax, (name, r), c in zip(axes, res.items(), colors):
        ax.scatter(yte, r['preds'], alpha=0.35, s=18, color=c)
        lim = max(yte.max(), r['preds'].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', lw=2, label='Idéal')
        ax.set_title(f"{name} — R²={r['R2']:.4f}", fontweight='bold')
        ax.set_xlabel('Prix Réel ($)'); ax.set_ylabel('Prix Prédit ($)')
        ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Résidus
    st.markdown('<div class="section-title">Analyse des Résidus</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (name, r), c in zip(axes, res.items(), colors):
        resid = yte.values - r['preds']
        ax.scatter(r['preds'], resid, alpha=0.35, s=15, color=c)
        ax.axhline(0, color='red', lw=2, ls='--')
        ax.set_title(f"Résidus — {name}", fontweight='bold')
        ax.set_xlabel('Prix Prédit ($)'); ax.set_ylabel('Résidu ($)')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature Importance
    st.markdown('<div class="section-title">Importance des Features — Random Forest</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    fi = M['fi_r']
    bars = ax.bar(fi.index, fi.values, color='#1a73e8', edgecolor='white', alpha=0.85)
    ax.set_ylabel('Importance'); ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, fi.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.2f}', ha='center', fontsize=8.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────
#  PAGE : CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────
elif "Classification" in page:
    st.markdown("## 🏷️ Résultats — Classification : Prédiction de BldgType")

    res = M['clf_res']; classes = M['classes']

    best = res['Random Forest']
    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Accuracy — Random Forest</div>
        <div class="kpi-value">{best['Accuracy']:.2%}</div>
        <div class="kpi-sub">Exactitude globale</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">F1-Score (weighted) — Random Forest</div>
        <div class="kpi-value">{best['F1']:.4f}</div>
        <div class="kpi-sub">Moyenne pondérée par classe</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tableau Comparatif</div>', unsafe_allow_html=True)
    comp_clf = pd.DataFrame({
        'Modèle': list(res.keys()),
        'Accuracy': [f"{v['Accuracy']:.4f}" for v in res.values()],
        'F1-Score (weighted)': [f"{v['F1']:.4f}" for v in res.values()]
    })
    st.dataframe(comp_clf, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Matrices de Confusion</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (name, r), cmap in zip(axes, res.items(), ['Blues', 'Greens']):
        sns.heatmap(r['CM'], annot=True, fmt='d', cmap=cmap,
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, linecolor='white')
        ax.set_title(f"Matrice de Confusion — {name}", fontweight='bold')
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    tab_svm, tab_rf = st.tabs(["📋 Rapport SVM", "📋 Rapport Random Forest"])
    with tab_svm:
        st.code(res['SVM (RBF)']['report'])
    with tab_rf:
        st.code(res['Random Forest']['report'])

    st.markdown('<div class="section-title">Importance des Features — Random Forest</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fi = M['fi_c']
    ax.barh(fi.index[::-1], fi.values[::-1], color='#2e7d32', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Importance'); ax.set_title('Feature Importance', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────
#  PAGE : PRÉDICTEUR
# ─────────────────────────────────────────────────────────────────────
elif "Prédicteur" in page:
    st.markdown("## 🔮 Prédicteur Immobilier Interactif")
    st.markdown("Renseignez les caractéristiques d'un bien pour estimer son **prix** et son **type**.")

    neighborhoods = sorted(df['Neighborhood'].unique().tolist())
    house_styles   = sorted(df['HouseStyle'].unique().tolist())

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">🏗️ Caractéristiques Générales</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            overall_qual = st.slider("Qualité Générale (1-10)", 1, 10, 7)
            overall_cond = st.slider("Condition Générale (1-10)", 1, 10, 5)
            year_built   = st.number_input("Année de construction", 1872, 2010, 2000, step=1)
        with col2:
            year_remod   = st.number_input("Année de rénovation", 1950, 2010, 2005, step=1)
            neighborhood = st.selectbox("Quartier (Neighborhood)", neighborhoods)
            house_style  = st.selectbox("Style (HouseStyle)", house_styles)
        with col3:
            fireplaces   = st.slider("Cheminées", 0, 4, 1)
            bedroom      = st.slider("Chambres", 0, 8, 3)
            full_bath    = st.slider("Salles de bain", 0, 4, 2)

        st.markdown('<div class="section-title">📐 Surfaces & Équipements</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            gr_liv  = st.number_input("Surface habitable (sq ft)", 300, 6000, 1500, step=50)
            bsmt    = st.number_input("Surface sous-sol (sq ft)", 0, 3000, 800, step=50)
            lot     = st.number_input("Surface terrain (sq ft)", 1000, 215000, 10000, step=500)
        with col5:
            tot_rms = st.slider("Nombre total de pièces", 2, 15, 7)
            garage_cars = st.slider("Capacité garage (voitures)", 0, 4, 2)
            garage_area = st.number_input("Surface garage (sq ft)", 0, 1500, 450, step=25)
        with col6:
            pool    = st.number_input("Surface piscine (sq ft)", 0, 800, 0, step=10)
            st.markdown("")
            st.markdown("")

        submitted = st.form_submit_button("🚀 Prédire", use_container_width=True, type="primary")

    if submitted:
        # ── Prédiction prix ──
        reg_inp = {
            'GrLivArea': gr_liv, 'TotalBsmtSF': bsmt, 'LotArea': lot,
            'BedroomAbvGr': bedroom, 'FullBath': full_bath, 'TotRmsAbvGrd': tot_rms,
            'OverallQual': overall_qual, 'OverallCond': overall_cond,
            'YearBuilt': year_built, 'YearRemodAdd': year_remod,
            'Neighborhood': neighborhood, 'GarageCars': garage_cars,
            'GarageArea': garage_area, 'PoolArea': pool, 'Fireplaces': fireplaces
        }
        inp_r = pd.DataFrame([reg_inp])
        inp_r['Neighborhood'] = M['le_neigh'].transform([neighborhood])
        inp_r = pd.DataFrame(M['imp_r'].transform(inp_r[REG_FEATURES]), columns=REG_FEATURES)
        price = M['rf_r'].predict(inp_r)[0]

        # ── Prédiction type ──
        clf_inp = {
            'GrLivArea': gr_liv, 'TotRmsAbvGrd': tot_rms,
            'OverallQual': overall_qual, 'YearBuilt': year_built,
            'GarageCars': garage_cars, 'Neighborhood': neighborhood,
            'HouseStyle': house_style
        }
        inp_c = pd.DataFrame([clf_inp])
        inp_c['Neighborhood'] = M['le_nc'].transform([neighborhood])
        inp_c['HouseStyle']   = M['le_hs'].transform([house_style])
        inp_c = pd.DataFrame(M['imp_c'].transform(inp_c[CLF_FEATURES]), columns=CLF_FEATURES)
        bldg_enc  = M['rf_c'].predict(M['scaler'].transform(inp_c))[0]
        bldg_type = M['le_bt'].inverse_transform([bldg_enc])[0]
        bldg_label = BLDG_LABELS.get(bldg_type, bldg_type)

        # ── Affichage ──
        st.markdown("---")
        col_r, col_c = st.columns(2)
        with col_r:
            st.markdown(f"""<div class="result-price">
                <div style="font-size:0.9rem;color:#388e3c;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.5rem">💰 Prix Estimé</div>
                <div class="result-price-val">${price:,.0f}</div>
                <div style="color:#4caf50;font-size:0.85rem;margin-top:0.5rem">
                    Random Forest Regressor · R²={M['reg_res']['Random Forest']['R2']:.3f}
                </div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="result-type">
                <div style="font-size:0.9rem;color:#1565c0;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:0.5rem">🏠 Type de Bâtiment</div>
                <div class="result-type-val">{bldg_type}</div>
                <div style="color:#1976d2;font-size:1.1rem;margin-top:0.3rem">{bldg_label}</div>
                <div style="color:#90caf9;font-size:0.82rem;margin-top:0.4rem">
                    Random Forest Classifier · Accuracy={M['clf_res']['Random Forest']['Accuracy']:.1%}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Contexte marché ──
        st.markdown("---")
        st.markdown("### 📊 Contexte du marché")
        ca, cb, cc = st.columns(3)
        p_med = df['SalePrice'].median(); p_mean = df['SalePrice'].mean()
        pct = (price - p_med) / p_med * 100
        ca.metric("Prix médian du marché", f"${p_med:,.0f}")
        cb.metric("Prix moyen du marché", f"${p_mean:,.0f}")
        cc.metric("Positionnement", f"${price:,.0f}",
                  delta=f"{pct:+.1f}% vs médiane",
                  delta_color="normal")