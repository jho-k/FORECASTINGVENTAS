# Forecasting Ventas

Estructura del proyecto de Machine Learning para forecasting de ventas.

Estructura sugerida:

- data/
  - raw/          # Datos originales sin tocar (no versionar)
  - processed/    # Datos procesados y listos para análisis
- notebooks/      # Notebooks Jupyter (exploración, EDA, experimentos)
- src/
  - data/         # scripts de carga y limpieza
  - features/     # creación de features
  - models/       # entrenamiento y evaluación de modelos
  - visualization/# scripts para gráficas
  - app/          # app (Streamlit) para demo/visualización
- models/         # modelos entrenados (pesados) (ignorado por git)
- reports/
  - figures/      # figuras y reportes exportados
- scripts/        # scripts utilitarios y de pipeline
- talks/          # material para charlas, presentaciones
- docs/           # documentación adicional

Archivos importantes:

- requirements.txt  -> dependencias del proyecto
- .gitignore        -> archivos a ignorar en git

Cómo empezar:
1. Crear y activar un entorno virtual.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Abrir `notebooks/00-exploratory.ipynb` para empezar el EDA.
4. Ejecutar `streamlit run src/app/streamlit_app.py` para lanzar la demo.
