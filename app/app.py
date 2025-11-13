import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Obtener el directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'modelo_final.joblib'
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'inferencia_df_transformado.csv'

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="📊 Forecasting Ventas Nov 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
ACCENT_COLOR = "#f093fb"

# TASAS DE CAMBIO (EUR como base)
EXCHANGE_RATES = {
    "EUR": 1.0,
    "USD": 1.10,      # 1 EUR = 1.10 USD
    "PEN": 4.20       # 1 EUR = 4.20 PEN (Soles Peruanos)
}

# MONEDAS DISPONIBLES
CURRENCIES = {
    "EUR": {"symbol": "€", "name": "Euros", "code": "EUR"},
    "USD": {"symbol": "$", "name": "Dólares", "code": "USD"},
    "PEN": {"symbol": "S/.", "name": "Soles", "code": "PEN"}
}

# Estilos CSS personalizados
st.markdown("""
    <style>
        .header-title {
            color: #667eea;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .black-friday-row {
            background-color: #fff3cd !important;
        }
        .separator {
            margin: 2rem 0;
            border-top: 3px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGAR DATOS Y MODELO
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Carga el modelo y los datos de inferencia"""
    try:
        model = load(str(MODEL_PATH))
        df = pd.read_csv(str(DATA_PATH))
        df['fecha'] = pd.to_datetime(df['fecha'])
        return model, df
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o datos: {str(e)}")
        st.error(f"📁 Rutas buscadas:\n- Modelo: {MODEL_PATH}\n- Datos: {DATA_PATH}")
        st.stop()

model, df_base = load_model_and_data()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_productos():
    """Obtiene lista de productos únicos"""
    return sorted(df_base['nombre'].unique().tolist())

def apply_discount_to_df(df, discount_pct):
    """Aplica el descuento al precio de venta"""
    df = df.copy()
    df['precio_venta'] = df['precio_base'] * (1 - discount_pct)
    return df

def apply_competition_scenario(df, scenario):
    """Aplica el escenario de competencia"""
    df = df.copy()
    adjustment = 0
    if scenario == "Competencia -5%":
        adjustment = -0.05
    elif scenario == "Competencia +5%":
        adjustment = 0.05
    
    df['Amazon'] = df['Amazon'] * (1 + adjustment)
    df['Decathlon'] = df['Decathlon'] * (1 + adjustment)
    df['Deporvillage'] = df['Deporvillage'] * (1 + adjustment)
    
    return df

def recalculate_price_features(df):
    """Recalcula las variables dependientes de precio"""
    df = df.copy()
    df['descuento_porcentaje'] = (df['precio_base'] - df['precio_venta']) / df['precio_base']
    df['precio_competencia'] = df[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']
    return df

def convert_currency(value_eur, from_currency="EUR", to_currency="EUR"):
    """Convierte un valor de una moneda a otra usando EUR como base"""
    if from_currency not in EXCHANGE_RATES or to_currency not in EXCHANGE_RATES:
        return value_eur
    
    # Convertir a EUR si no lo es
    if from_currency != "EUR":
        value_eur = value_eur / EXCHANGE_RATES[from_currency]
    
    # Convertir de EUR a la moneda destino
    return value_eur * EXCHANGE_RATES[to_currency]

def format_currency_display(value_eur, currency_code="EUR"):
    """Formatea un valor monetario con la moneda especificada"""
    converted_value = convert_currency(value_eur, from_currency="EUR", to_currency=currency_code)
    symbol = CURRENCIES[currency_code]["symbol"]
    
    # Formatear según la moneda (PEN usa 0 decimales, USD y EUR usan 2)
    if currency_code == "PEN":
        return f"{symbol}{converted_value:,.0f}"
    else:
        return f"{symbol}{converted_value:,.2f}"

def get_feature_columns():
    """Obtiene las columnas de features esperadas por el modelo"""
    return model.feature_names_in_

def predict_recursive(df_product, model):
    """
    Realiza predicciones recursivas actualizando lags día a día
    """
    df = df_product.sort_values('fecha').reset_index(drop=True).copy()
    predictions = []
    dates = []
    
    feature_cols = get_feature_columns()
    
    # Guardar lags iniciales (del día 1)
    lag_history = []
    
    for idx, row in df.iterrows():
        # Para días después del primero, actualizar lags
        if idx > 0:
            pred_anterior = predictions[-1]
            
            # Desplazar lags
            new_lags = [pred_anterior]  # lag_1 = predicción del día anterior
            if len(lag_history) > 0:
                new_lags.extend(lag_history[idx-1][:6])  # agregar los 6 lags anteriores
            
            lag_history.append(new_lags)
            
            # Actualizar los valores de lag en el dataframe
            for lag_idx, lag_val in enumerate(new_lags):
                lag_col = f'unidades_vendidas_lag{lag_idx + 1}'
                if lag_col in df.columns:
                    df.loc[idx, lag_col] = lag_val
            
            # Actualizar media móvil con las últimas 7 predicciones
            recent_preds = predictions[max(0, len(predictions)-7):]
            df.loc[idx, 'unidades_vendidas_ma7'] = np.mean(recent_preds)
        else:
            # Guardar lags iniciales del primer día
            lag_vals = [df.loc[idx, f'unidades_vendidas_lag{i}'] for i in range(1, 8)]
            lag_history.append(lag_vals)
        
        # Preparar features para predicción
        X = df.loc[[idx], feature_cols].values
        
        # Realizar predicción
        pred = model.predict(X)[0]
        pred = max(0, pred)  # Asegurar que no sea negativo
        
        predictions.append(pred)
        dates.append(row['fecha'])
    
    return predictions, dates

# ============================================================================
# SIDEBAR - CONTROLES DE SIMULACIÓN
# ============================================================================

with st.sidebar:
    st.markdown("## 🎮 Controles de Simulación")
    
    # Selector de moneda
    st.markdown("### 💱 Selecciona tu Moneda")
    currency_option = st.radio(
        "Moneda:",
        list(CURRENCIES.keys()),
        format_func=lambda x: f"{CURRENCIES[x]['symbol']} {CURRENCIES[x]['name']}",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Selector de producto
    producto_seleccionado = st.selectbox(
        "📦 Selecciona un producto:",
        get_productos(),
        help="Elige el producto para hacer la simulación"
    )
    
    st.divider()
    
    # Slider de descuento
    discount_slider = st.slider(
        "💰 Ajuste de descuento:",
        min_value=-50,
        max_value=50,
        step=5,
        value=0,
        format="%d%%",
        help="Rango: -50% a +50%"
    )
    discount_pct = discount_slider / 100  # Convertir a decimal
    
    st.divider()
    
    # Selector de escenario de competencia
    scenario = st.radio(
        "🏪 Escenario de competencia:",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Cómo varían los precios de la competencia"
    )
    
    st.divider()
    
    # Botón de simulación
    simulate_btn = st.button(
        "▶️ SIMULAR VENTAS",
        use_container_width=True,
        type="primary"
    )

# ============================================================================
# LÓGICA PRINCIPAL
# ============================================================================

if simulate_btn:
    with st.spinner("⏳ Procesando predicciones recursivas..."):
        # 1. Filtrar por producto
        df_product = df_base[df_base['nombre'] == producto_seleccionado].copy()
        
        if df_product.empty:
            st.error("❌ No hay datos disponibles para este producto")
            st.stop()
        
        # 2. Aplicar descuento
        df_product = apply_discount_to_df(df_product, discount_pct)
        
        # 3. Aplicar escenario de competencia
        df_product = apply_competition_scenario(df_product, scenario)
        
        # 4. Recalcular variables de precio
        df_product = recalculate_price_features(df_product)
        
        # 5. Realizar predicciones recursivas
        predictions, dates = predict_recursive(df_product, model)
        
        # Guardar en session state
        st.session_state.predictions = predictions
        st.session_state.dates = dates
        st.session_state.df_product = df_product.sort_values('fecha').reset_index(drop=True)
        st.session_state.producto_seleccionado = producto_seleccionado
        st.session_state.discount_pct = discount_pct
        st.session_state.scenario = scenario

# ============================================================================
# ÁREA PRINCIPAL - DASHBOARD
# ============================================================================

if 'predictions' in st.session_state:
    predictions = st.session_state.predictions
    dates = st.session_state.dates
    df_product = st.session_state.df_product
    producto_seleccionado = st.session_state.producto_seleccionado
    
    st.markdown(
        f"<div class='header-title'>📊 Dashboard Predicción de Ventas</div>",
        unsafe_allow_html=True
    )
    discount_display = f"{st.session_state.discount_pct*100:+.0f}%"
    st.markdown(
        f"**Producto:** {producto_seleccionado} | **Mes:** Noviembre 2025 | "
        f"**Descuento:** {discount_display} | **Escenario:** {st.session_state.scenario}",
        unsafe_allow_html=True
    )
    
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    
    # KPIs
    st.subheader("📈 KPIs Proyectados")
    
    total_unidades = sum(predictions)
    df_product_sorted = df_product.sort_values('fecha').reset_index(drop=True)
    df_product_sorted['prediccion'] = predictions
    df_product_sorted['ingresos'] = df_product_sorted['prediccion'] * df_product_sorted['precio_venta']
    total_ingresos = df_product_sorted['ingresos'].sum()
    precio_promedio = df_product_sorted['precio_venta'].mean()
    descuento_promedio = df_product_sorted['descuento_porcentaje'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Unidades Totales",
            f"{total_unidades:,.0f}",
            delta=None
        )
    
    with col2:
        ingresos_formateado = format_currency_display(total_ingresos, currency_option)
        st.metric(
            "� Ingresos Totales",
            ingresos_formateado,
            delta=None
        )
    
    with col3:
        precio_formateado = format_currency_display(precio_promedio, currency_option)
        st.metric(
            "� Precio Promedio",
            precio_formateado,
            delta=None
        )
    
    with col4:
        descuento_display_metric = f"{descuento_promedio*100:+.2f}%"
        st.metric(
            "🏷️ Descuento Promedio",
            descuento_display_metric,
            delta=None
        )
    
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    
    # GRÁFICO DE PREDICCIÓN DIARIA
    st.subheader("📉 Predicción de Ventas Diarias (Noviembre)")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Datos para gráfico
    dias_num = [d.day for d in dates]
    
    # Línea de predicción
    ax.plot(dias_num, predictions, 
            color=PRIMARY_COLOR, linewidth=3, marker='o', 
            markersize=6, label='Predicción de Ventas', zorder=2)
    
    # Área bajo la línea
    ax.fill_between(dias_num, predictions, alpha=0.2, color=PRIMARY_COLOR)
    
    # Marcar Black Friday (día 28)
    bf_idx = [i for i, d in enumerate(dias_num) if d == 28]
    if bf_idx:
        bf_idx = bf_idx[0]
        ax.axvline(x=28, color='#ff6b6b', linestyle='--', linewidth=2.5, 
                   alpha=0.7, label='Black Friday', zorder=1)
        ax.plot(28, predictions[bf_idx], 'o', color='#ff6b6b', 
               markersize=12, zorder=3)
        ax.annotate('🎉 BLACK FRIDAY 🎉', 
                   xy=(28, predictions[bf_idx]),
                   xytext=(26, predictions[bf_idx] * 1.15),
                   fontsize=11, fontweight='bold',
                   color='#ff6b6b',
                   arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2),
                   ha='center')
    
    # Estilos
    ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Predichas', fontsize=12, fontweight='bold')
    ax.set_title('Evolución de Ventas Proyectadas', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks(range(1, 31, 2))
    
    # Colores de fondo
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    
    # TABLA DETALLADA
    st.subheader("📋 Detalle Diario de Noviembre")
    
    df_tabla = df_product_sorted[[
        'fecha', 'dia_semana', 'precio_venta', 
        'precio_competencia', 'descuento_porcentaje'
    ]].copy()
    df_tabla['prediccion'] = predictions
    df_tabla['ingresos_diarios'] = df_tabla['prediccion'] * df_tabla['precio_venta']
    
    # Renombrar columnas para la tabla
    moneda_symbol = CURRENCIES[currency_option]["symbol"]
    df_tabla.columns = [
        'Fecha', 'Día Semana', f'Precio Venta ({moneda_symbol})', 
        f'Competencia ({moneda_symbol})', 'Descuento', 'Unidades', f'Ingresos ({moneda_symbol})'
    ]
    
    # Formatear con conversión de moneda
    df_tabla['Fecha'] = df_tabla['Fecha'].dt.strftime('%d/%m/%Y')
    df_tabla[f'Precio Venta ({moneda_symbol})'] = df_tabla[f'Precio Venta ({moneda_symbol})'].apply(
        lambda x: format_currency_display(x, currency_option)
    )
    df_tabla[f'Competencia ({moneda_symbol})'] = df_tabla[f'Competencia ({moneda_symbol})'].apply(
        lambda x: format_currency_display(x, currency_option)
    )
    df_tabla['Descuento'] = df_tabla['Descuento'].apply(lambda x: f'{x*100:+.2f}%')
    df_tabla['Unidades'] = df_tabla['Unidades'].apply(lambda x: f'{x:.0f}')
    df_tabla[f'Ingresos ({moneda_symbol})'] = df_tabla[f'Ingresos ({moneda_symbol})'].apply(
        lambda x: format_currency_display(x, currency_option)
    )
    
    # Mostrar tabla con highlight para Black Friday
    st.dataframe(
        df_tabla.style.applymap(
            lambda x: 'background-color: #fff3cd; font-weight: bold;' 
            if '28/11/2025' in str(x) else '',
            subset=['Fecha']
        ),
        use_container_width=True,
        height=600
    )
    
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    
    # COMPARATIVA DE ESCENARIOS
    st.subheader("🔄 Comparativa de Escenarios de Competencia")
    st.markdown("*Manteniendo el descuento del usuario, solo varía el escenario de competencia*")
    
    # Calcular predicciones para los 3 escenarios
    scenarios_data = {}
    
    for scenario_name in ["Actual (0%)", "Competencia -5%", "Competencia +5%"]:
        df_temp = df_base[df_base['nombre'] == producto_seleccionado].copy()
        df_temp = apply_discount_to_df(df_temp, st.session_state.discount_pct)
        df_temp = apply_competition_scenario(df_temp, scenario_name)
        df_temp = recalculate_price_features(df_temp)
        
        preds, _ = predict_recursive(df_temp, model)
        
        scenarios_data[scenario_name] = {
            'unidades': sum(preds),
            'ingresos': sum(preds * df_temp.sort_values('fecha')['precio_venta'].values)
        }
    
    # Mostrar tarjetas de escenarios
    cols = st.columns(3)
    
    for col_idx, (scenario_name, data) in enumerate(scenarios_data.items()):
        with cols[col_idx]:
            col_color = PRIMARY_COLOR if scenario_name == st.session_state.scenario else SECONDARY_COLOR
            ingresos_convertidos = format_currency_display(data["ingresos"], currency_option)
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, {col_color} 0%, {SECONDARY_COLOR} 100%);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    border: 3px solid {col_color if scenario_name == st.session_state.scenario else "transparent"};
                '>
                    <h3 style='margin: 0 0 1rem 0;'>{scenario_name}</h3>
                    <p style='margin: 0.5rem 0; font-size: 1.2rem;'><strong>{data["unidades"]:,.0f}</strong> unidades</p>
                    <p style='margin: 0.5rem 0; font-size: 1.2rem;'><strong>{ingresos_convertidos}</strong> ingresos</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr class='separator'>", unsafe_allow_html=True)
    
    # INFORMACIÓN ADICIONAL
    with st.expander("ℹ️ Información Técnica"):
        st.markdown("""
        **Metodología:**
        - Predicciones recursivas actualizando lags día a día
        - Lags iniciales del 1 de noviembre calculados desde octubre
        - Media móvil de 7 días actualizada con las últimas predicciones
        - Modelo: HistGradientBoostingRegressor
        
        **Variables ajustadas por simulación:**
        - Precio de venta: según descuento aplicado
        - Precios de competencia: según escenario
        - Ratio precio y descuento porcentual: recalculados automáticamente
        """)

else:
    # Estado inicial
    st.markdown("""
        <div style='text-align: center; padding: 3rem; color: #667eea;'>
            <h2>👋 Bienvenido al Dashboard de Forecasting</h2>
            <p style='font-size: 1.1rem; color: #764ba2;'>
                Selecciona un producto en el sidebar y haz clic en 
                <strong>"SIMULAR VENTAS"</strong> para comenzar
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.info("""
    **📌 Instrucciones:**
    1. En el panel izquierdo, selecciona el producto deseado
    2. Ajusta el descuento con el slider (-50% a +50%)
    3. Elige un escenario de competencia
    4. Haz clic en "SIMULAR VENTAS"
    5. Visualiza predicciones, gráficos y comparativas
    """)
