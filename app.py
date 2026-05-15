import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Предсказание цен на недвижимость")
st.markdown("*ML-сервис для автоматического предсказания стоимости домов*")
st.markdown("---")

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================
@st.cache_resource
def load_model():
    model_files = ['best_model_optuna.pkl', 'best_model_balanced.pkl', 'best_model.pkl']
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            st.success(f"✅ Модель загружена: {model_file}")
            return model, model_file
        except FileNotFoundError:
            continue
    st.error("❌ Модель не найдена!")
    return None, None

model, model_name = load_model()

def get_model_type():
    if model is None:
        return "Не загружена"
    try:
        model_str = str(type(model.named_steps['regressor'])).lower()
        if 'xgb' in model_str:
            return "🚀 XGBoost (Optuna оптимизация)"
        elif 'randomforest' in model_str:
            return "🌲 RandomForest"
        else:
            return "🤖 Оптимизированная модель"
    except:
        return "Модель загружена"

# ============================================================
# ФУНКЦИИ ОБРАБОТКИ ДАННЫХ (те же, что ранее)
# ============================================================
def safe_fill_missing_values(df):
    df = df.copy()
    cat_cols_fill_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                          'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
                          'MasVnrType']
    for col in cat_cols_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    num_cols_fill_median = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                            'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']
    for col in num_cols_fill_median:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    if 'MasVnrArea' in df.columns:
        if 'MasVnrType' in df.columns:
            df.loc[df['MasVnrType'] == 'None', 'MasVnrArea'] = 0
        df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
    
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    if 'GarageYrBlt' in df.columns:
        if 'GarageArea' in df.columns and 'GarageType' in df.columns:
            no_garage_mask = (df['GarageArea'] == 0) | (df['GarageType'] == 'None')
            df.loc[no_garage_mask, 'GarageYrBlt'] = df.loc[no_garage_mask, 'GarageYrBlt'].fillna(0)
        if 'YearBuilt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    
    defaults = {
        'MSZoning': 'RL', 'Utilities': 'AllPub', 'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd', 'KitchenQual': 'TA', 'Functional': 'Typ',
        'SaleType': 'WD', 'Electrical': 'SBrkr'
    }
    for col, default_val in defaults.items():
        if col in df.columns:
            mode_vals = df[col].mode()
            df[col] = df[col].fillna(mode_vals[0] if len(mode_vals) > 0 else default_val)
    return df

def safe_feature_engineering(df):
    df = df.copy()
    # TotalSF
    total_sf = 0
    if 'TotalBsmtSF' in df:
        total_sf += df['TotalBsmtSF'].fillna(0)
    if '1stFlrSF' in df:
        total_sf += df['1stFlrSF'].fillna(0)
    if '2ndFlrSF' in df:
        total_sf += df['2ndFlrSF'].fillna(0)
    df['TotalSF'] = total_sf
    
    # TotalBath
    total_bath = 0
    if 'FullBath' in df:
        total_bath += df['FullBath'].fillna(0)
    if 'HalfBath' in df:
        total_bath += 0.5 * df['HalfBath'].fillna(0)
    if 'BsmtFullBath' in df:
        total_bath += df['BsmtFullBath'].fillna(0)
    if 'BsmtHalfBath' in df:
        total_bath += 0.5 * df['BsmtHalfBath'].fillna(0)
    df['TotalBath'] = total_bath
    
    # TotalPorchSF
    total_porch = 0
    for col in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
        if col in df:
            total_porch += df[col].fillna(0)
    df['TotalPorchSF'] = total_porch
    
    df['HasPool'] = (df['PoolArea'].fillna(0) > 0).astype(int) if 'PoolArea' in df else 0
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int) if 'GarageArea' in df else 0
    df['HasBsmt'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int) if 'TotalBsmtSF' in df else 0
    
    if 'YrSold' in df and 'YearBuilt' in df:
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    else:
        df['Age'] = 0
        df['IsNew'] = 0
    if 'YrSold' in df and 'YearRemodAdd' in df:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    else:
        df['RemodAge'] = 0
    return df

# ============================================================
# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ
# ============================================================
if 'predictions_ready' not in st.session_state:
    st.session_state.predictions_ready = False
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'prev_predictions' not in st.session_state:
    st.session_state.prev_predictions = None

# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================
with st.sidebar:
    st.header("📊 О сервисе")
    st.markdown(f"**Модель:** {get_model_type()}")
    st.markdown(f"**Файл:** `{model_name if model_name else '—'}`")
    st.markdown("**Метрика:** RMSLE (~0.125–0.127)")
    st.markdown("---")
    st.header("📁 Формат файла")
    st.markdown("CSV файл как `test.csv` (все исходные признаки)")
    st.markdown("---")
    st.caption("© 2025 House Price Prediction")

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================================
uploaded_file = st.file_uploader("📂 Выберите CSV файл с данными", type=['csv'])

if uploaded_file is not None and model is not None:
    # Загружаем данные
    input_data = pd.read_csv(uploaded_file)
    st.session_state.input_data = input_data
    st.success(f"✅ Загружено {input_data.shape[0]} строк, {input_data.shape[1]} колонок")
    with st.expander("🔍 Превью данных"):
        st.dataframe(input_data.head())
    
    # Кнопка предсказания
    if st.button("🔮 Предсказать цены", type="primary", use_container_width=True):
        with st.spinner("Обработка и предсказание..."):
            processed = safe_fill_missing_values(input_data)
            processed = safe_feature_engineering(processed)
            log_pred = model.predict(processed)
            predictions = np.expm1(log_pred)
            st.session_state.current_predictions = predictions
            st.session_state.predictions_ready = True
            st.rerun()  # Принудительно перезапускаем, чтобы отобразить результаты

# Отображение результатов, если они уже есть
if st.session_state.predictions_ready and st.session_state.current_predictions is not None:
    predictions = st.session_state.current_predictions
    input_data = st.session_state.input_data
    
    st.subheader("📊 Результаты предсказания")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Средняя цена", f"${predictions.mean():,.0f}")
    col2.metric("📊 Медиана", f"${np.median(predictions):,.0f}")
    col3.metric("📈 Минимум", f"${predictions.min():,.0f}")
    col4.metric("📉 Максимум", f"${predictions.max():,.0f}")
    
    # Вкладки
    tab1, tab2, tab3 = st.tabs(["📊 Распределение цен", "🔍 Важность признаков", "📈 Сравнение моделей"])
    
    with tab1:
        fig_hist = px.histogram(predictions, nbins=50, 
                                title="Распределение предсказанных цен",
                                labels={'value': 'Цена ($)', 'count': 'Частота'},
                                color_discrete_sequence=['#1f77b4'])
        fig_hist.add_vline(x=predictions.mean(), line_dash="dash", line_color="red",
                           annotation_text=f"Среднее: ${predictions.mean():,.0f}")
        fig_hist.add_vline(x=np.median(predictions), line_dash="dash", line_color="green",
                           annotation_text=f"Медиана: ${np.median(predictions):,.0f}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        stats_df = pd.DataFrame({
            "Метрика": ["Среднее", "Медиана", "Станд. отклонение", "Q1", "Q3", "Min", "Max"],
            "Значение ($)": [
                f"{predictions.mean():,.0f}", f"{np.median(predictions):,.0f}",
                f"{predictions.std():,.0f}", f"{np.percentile(predictions, 25):,.0f}",
                f"{np.percentile(predictions, 75):,.0f}", f"{predictions.min():,.0f}",
                f"{predictions.max():,.0f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
        st.markdown("""
**📖 Что показывает этот график:**  
- Гистограмма отображает, как распределяются предсказанные цены домов.  
- Красная пунктирная линия — средняя цена, зелёная — медианная.  
- Если гистограмма симметрична и похожа на колокол — это хорошо.  
- Значительный перекос вправо означает много дорогих домов, влево — много дешёвых.
""")
    
    with tab2:
        try:
            regressor = model.named_steps['regressor']
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
                indices = np.argsort(importances)[-15:][::-1]
                top_importances = importances[indices]
                feature_names = [f"Признак {i}" for i in indices]
                if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                    try:
                        all_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                        feature_names = [all_feature_names[i] for i in indices if i < len(all_feature_names)]
                    except:
                        pass
                fig_imp = go.Figure(go.Bar(x=top_importances, y=feature_names, orientation='h'))
                fig_imp.update_layout(title="Топ-15 важности признаков", 
                                      xaxis_title="Важность", yaxis_title="Признак")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Модель не предоставляет важность признаков.")
        except Exception as e:
            st.warning(f"Не удалось отобразить важность признаков: {e}")
        st.markdown("""
**📖 Что показывают важности признаков:**  
- Длина полоски — вклад признака в предсказание цены.  
- Самые важные признаки (обычно: общее качество, жилая площадь, возраст дома) должны соответствовать логике рынка недвижимости.  
- Если какой-то странный признак оказался в топе — возможно, модель переобучена.
""")
    
    with tab3:
        # Загрузка предыдущего файла
        prev_file = st.file_uploader("Загрузите предыдущий CSV (Id, SalePrice)", type=['csv'], key="prev_comp")
        if prev_file is not None:
            prev_df = pd.read_csv(prev_file)
            if 'SalePrice' in prev_df.columns:
                prev_preds = prev_df['SalePrice'].values
                st.session_state.prev_predictions = prev_preds
                st.success("Файл загружен.")
            else:
                st.warning("Файл должен содержать колонку 'SalePrice'.")
        
        # Сравнение, если есть оба
        if st.session_state.prev_predictions is not None:
            prev = st.session_state.prev_predictions
            curr = predictions
            if len(prev) == len(curr):
                diff = curr - prev
                fig_comp = px.scatter(x=prev, y=curr, 
                                      labels={'x': 'Предыдущая модель ($)', 'y': 'Текущая модель ($)'},
                                      title="Сравнение предсказаний двух моделей")
                fig_comp.add_shape(type='line', x0=0, y0=0, x1=600000, y1=600000,
                                   line=dict(dash='dash', color='red'))
                st.plotly_chart(fig_comp, use_container_width=True)
                st.metric("Среднее абсолютное изменение", f"${np.abs(diff).mean():,.0f}")
                st.metric("Корреляция", f"{np.corrcoef(prev, curr)[0,1]:.4f}")
            else:
                st.warning("Количество предсказаний не совпадает.")
        else:
            st.info("Загрузите CSV файл с предсказаниями предыдущей модели.")
        st.markdown("""
**📖 Интерпретация графика сравнения:**  
- Каждая точка — один дом.  
- Красная линия \(y = x\) — идеальное совпадение двух моделей.  
- Если точки разбросаны далеко от линии — модели сильно расходятся.  
- Большое среднее абсолютное изменение говорит о том, что модели дают систематически разные оценки.
""")
    
    # Кнопка скачивания результатов
    results = input_data.copy()
    results['Predicted_Price'] = predictions
    csv = results.to_csv(index=False)
    st.download_button(
        label="💾 Скачать предсказания (CSV)",
        data=csv,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    if model is None:
        st.warning("Модель не загружена. Поместите файл best_model_optuna.pkl в директорию.")
    else:
        st.info("Загрузите CSV файл с признаками домов и нажмите «Предсказать цены».")

st.markdown("---")
st.caption(f"🤖 Модель: {get_model_type()} | 📁 Используйте test.csv или аналогичный формат")