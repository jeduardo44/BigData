from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when, expr, lit, sum as spark_sum, count as spark_count
from pyspark.sql.types import IntegerType, FloatType, DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, stddev, col
import pyspark.sql.functions as F

spark = SparkSession \
.builder \
.master("yarn") \
.config("spark.ui.port", "4041") \
.config("spark.executor.memory", "4g") \
.config("spark.driver.memory", "4g") \
.getOrCreate()

# Carregar os dados
flights_df = spark.read.csv("path", header=True, inferSchema=True)
airlines_df = spark.read.csv("path", header=True, inferSchema=True)
airports_df = spark.read.csv("path", header=True, inferSchema=True)

# Selecionar 10% dos dados para teste
sampled_flights_df = flights_df.sample(fraction=0.25, seed=42)
print(f"Dados originais: {flights_df.count()} linhas")
print(f"Amostra de 25%: {sampled_flights_df.count()} linhas")

# Joins com airlines e airports
airlines_df = airlines_df \
    .withColumnRenamed("AIRLINE", "AIRLINE_NAME") \
    .withColumnRenamed("IATA_CODE", "AIRLINE_IATA_CODE")

sampled_flights_df = sampled_flights_df.join(
    airlines_df,
    sampled_flights_df.AIRLINE == airlines_df.AIRLINE_IATA_CODE,
    "left"
)

origin_airports = airports_df.alias("origin")
sampled_flights_df = sampled_flights_df.join(
    origin_airports,
    sampled_flights_df.ORIGIN_AIRPORT == origin_airports.IATA_CODE,
    "left"
) \
.select(
    sampled_flights_df['*'],
    col("origin.AIRPORT").alias("ORIGIN_AIRPORT_NAME"),
    col("origin.CITY").alias("ORIGIN_CITY"),
    col("origin.STATE").alias("ORIGIN_STATE"),
    col("origin.COUNTRY").alias("ORIGIN_COUNTRY"),
    col("origin.LATITUDE").alias("ORIGIN_LATITUDE"),
    col("origin.LONGITUDE").alias("ORIGIN_LONGITUDE")
)

dest_airports = airports_df.alias("dest")
sampled_flights_df = sampled_flights_df.join(
    dest_airports,
    sampled_flights_df.DESTINATION_AIRPORT == dest_airports.IATA_CODE,
    "left"
) \
.select(
    sampled_flights_df['*'],
    col("dest.AIRPORT").alias("DEST_AIRPORT_NAME"),
    col("dest.CITY").alias("DEST_CITY"),
    col("dest.STATE").alias("DEST_STATE"),
    col("dest.COUNTRY").alias("DEST_COUNTRY"),
    col("dest.LATITUDE").alias("DEST_LATITUDE"),
    col("dest.LONGITUDE").alias("DEST_LONGITUDE")
)

print(f"Total de registros após os joins: {sampled_flights_df.count()}")


# 2. Tratamento de valores faltantes
df_cleaned = sampled_flights_df

# Tratamento para colunas numéricas
numeric_cols = [
    "DEPARTURE_DELAY", "ARRIVAL_DELAY", "ELAPSED_TIME",
    "AIR_TIME", "DISTANCE", "ORIGIN_LATITUDE", "ORIGIN_LONGITUDE",
    "DEST_LATITUDE", "DEST_LONGITUDE"
]

# Calcular estatísticas para detecção de outliers
stats = {}
for nc in numeric_cols:
    stats[nc] = df_cleaned.select(
        avg(col(nc)).alias("mean"),
        stddev(col(nc)).alias("stddev"),
        F.expr(f"percentile({nc}, 0.25)").alias("q1"),
        F.expr(f"percentile({nc}, 0.5)").alias("median"),
        F.expr(f"percentile({nc}, 0.75)").alias("q3")
    ).collect()[0]

# Tratar valores faltantes para cada coluna
for nc in numeric_cols:
    # Substituir valores nulos pela mediana
    df_cleaned = df_cleaned.withColumn(
        nc,
        when(col(nc).isNull(), stats[nc]["median"]).otherwise(col(nc))
    )

    # Tratar outliers usando IQR (Intervalo Interquartil)
    iqr = stats[nc]["q3"] - stats[nc]["q1"]
    lower_bound = stats[nc]["q1"] - 1.5 * iqr
    upper_bound = stats[nc]["q3"] + 1.5 * iqr

    # Substituir outliers pelos limites
    df_cleaned = df_cleaned.withColumn(
        nc,
        when(col(nc) < lower_bound, lower_bound)
        .when(col(nc) > upper_bound, upper_bound)
        .otherwise(col(nc))
    )

# Para colunas categóricas, substituir valores nulos com "UNKNOWN"
categorical_cols = [
    "AIRLINE", "AIRLINE_NAME", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "ORIGIN_AIRPORT_NAME", "ORIGIN_CITY", "ORIGIN_STATE", "ORIGIN_COUNTRY",
    "DEST_AIRPORT_NAME", "DEST_CITY", "DEST_STATE", "DEST_COUNTRY"
]

for cc in categorical_cols:
    df_cleaned = df_cleaned.withColumn(
        cc,
        when(col(cc).isNull(), "UNKNOWN").otherwise(col(cc))
    )

# 3. Feature Engineering

# Criar novas features
df_cleaned = df_cleaned.withColumn("DELAY_RATIO",
                                   when(col("SCHEDULED_TIME") > 0,
                                        col("ARRIVAL_DELAY") / col("SCHEDULED_TIME"))
                                   .otherwise(0))

# Calcular distância haversine entre aeroportos de origem e destino
df_cleaned = df_cleaned.withColumn(
    "HAVERSINE_DISTANCE",
    expr("""
    2 * 6371 * asin(
        sqrt(
            pow(sin(radians(DEST_LATITUDE - ORIGIN_LATITUDE) / 2), 2) +
            cos(radians(ORIGIN_LATITUDE)) * cos(radians(DEST_LATITUDE)) *
            pow(sin(radians(DEST_LONGITUDE - ORIGIN_LONGITUDE) / 2), 2)
        )
    )
    """)
)

# Criar features de dia da semana e mês
df_cleaned = df_cleaned.withColumn("DAY_OF_WEEK",
                                   col("DAY_OF_WEEK").cast(IntegerType()))
df_cleaned = df_cleaned.withColumn("MONTH",
                                   col("MONTH").cast(IntegerType()))

# Criar features de hora do dia para partida e chegada
df_cleaned = df_cleaned.withColumn("DEPARTURE_HOUR",
                                   (col("SCHEDULED_DEPARTURE") / 100).cast(IntegerType()))
df_cleaned = df_cleaned.withColumn("ARRIVAL_HOUR",
                                   (col("SCHEDULED_ARRIVAL") / 100).cast(IntegerType()))

# 4. Transformação de variáveis categóricas
# Preparar pipeline para transformação de categóricas
categorical_features = []
string_indexers = []
one_hot_encoders = []
feature_names = []

for categorical_col in categorical_cols:
    indexer_name = f"{categorical_col}_indexer"
    encoder_name = f"{categorical_col}_encoder"
    output_name = f"{categorical_col}_encoded"

    # Criar o indexador para cada coluna categórica
    indexer = StringIndexer(
        inputCol=categorical_col,
        outputCol=indexer_name,
        handleInvalid="keep"
    )

    # Criar o encoder One-Hot para cada coluna indexada
    encoder = OneHotEncoder(
        inputCol=indexer_name,
        outputCol=output_name
    )

    string_indexers.append(indexer)
    one_hot_encoders.append(encoder)
    categorical_features.append(output_name)
    feature_names.append(output_name)

# 5. Criar buckets para variáveis numéricas
# Exemplo: criar buckets para DISTANCE
distance_bucketizer = Bucketizer(
    splits=[0, 500, 1000, 1500, 2000, float('Inf')],
    inputCol="DISTANCE",
    outputCol="DISTANCE_BUCKET"
)

# Adicionar bucketizer ao pipeline e feature names
feature_names.append("DISTANCE_BUCKET")

# 6. Normalizar features numéricas
numerical_features = [
    "DEPARTURE_DELAY", "ARRIVAL_DELAY", "ELAPSED_TIME",
    "AIR_TIME", "DISTANCE", "HAVERSINE_DISTANCE",
    "DEPARTURE_HOUR", "ARRIVAL_HOUR", "DELAY_RATIO",
    "DAY_OF_WEEK", "MONTH"
]

for num_col in numerical_features:
    feature_names.append(num_col)

# 7. Montar o Vector Assembler final
assembler = VectorAssembler(
    inputCols=feature_names,
    outputCol="features",
    handleInvalid="keep"
)

# Opcional: adicionar StandardScaler para normalizar todas as features numéricas
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

# 8. Montar o pipeline completo
pipeline_stages = string_indexers + one_hot_encoders + [distance_bucketizer, assembler, scaler]
feature_pipeline = Pipeline(stages=pipeline_stages)

# 9. Aplicar o pipeline aos dados
print("Aplicando pipeline de feature engineering...")
feature_model = feature_pipeline.fit(df_cleaned)
transformed_df = feature_model.transform(df_cleaned)

# 11. Preparar dados para treinamento do modelo
final_df = transformed_df.select(
    col("ARRIVAL_DELAY").alias("label"),
    col("scaled_features")
)

# Adicionar verificação de dados duplicados ou problemáticos
duplicate_count = transformed_df.count() - transformed_df.dropDuplicates().count()
print(f"Número de registros duplicados: {duplicate_count}")

# Verificar valores extremos após transformação
print("Estatísticas dos dados transformados:")
final_df.describe("label").show()

# Preparar os dados para os diferentes modelos
print("Preparando dados para modelos de classificação e regressão...")

# 1. Criar variáveis target para classificação (se vai atrasar ou não)
threshold_minutes = 0  # Definir atraso como > 0 minutos

df_with_targets = transformed_df.withColumn(
    "DEPARTURE_DELAYED",
    when(col("DEPARTURE_DELAY") > threshold_minutes, 1.0).otherwise(0.0)
).withColumn(
    "ARRIVAL_DELAYED",
    when(col("ARRIVAL_DELAY") > threshold_minutes, 1.0).otherwise(0.0)
)

# 2. Separar em conjuntos de treino e teste
train_data, test_data = df_with_targets.randomSplit([0.8, 0.2], seed=42)
print(f"Dados de treino: {train_data.count()} registros")
print(f"Dados de teste: {test_data.count()} registros")

# 3. Definir função para criar e avaliar modelos
def train_and_evaluate_models(train_df, test_df):
    results = {}

    # Features já processadas do pipeline anterior
    feature_col = "scaled_features"

    # ===== MODELOS DE CLASSIFICAÇÃO =====

    # Modelo para prever DEPARTURE_DELAYED
    lr_departure = LogisticRegression(
        featuresCol=feature_col,
        labelCol="DEPARTURE_DELAYED",
        maxIter=10
    )

    # Modelo para prever ARRIVAL_DELAYED
    lr_arrival = LogisticRegression(
        featuresCol=feature_col,
        labelCol="ARRIVAL_DELAYED",
        maxIter=10
    )

    # Treinar modelos de classificação
    print("Treinando modelos de classificação...")
    lr_departure_model = lr_departure.fit(train_df)
    lr_arrival_model = lr_arrival.fit(train_df)

    # Avaliar modelos de classificação
    departure_predictions = lr_departure_model.transform(test_df)
    arrival_predictions = lr_arrival_model.transform(test_df)

    evaluator_class = BinaryClassificationEvaluator(metricName="areaUnderROC")

    # Configurar avaliador corretamente para Spark
    evaluator_class.setLabelCol("DEPARTURE_DELAYED")
    departure_auc = evaluator_class.evaluate(departure_predictions)

    evaluator_class.setLabelCol("ARRIVAL_DELAYED")
    arrival_auc = evaluator_class.evaluate(arrival_predictions)

    results["departure_classification_auc"] = departure_auc
    results["arrival_classification_auc"] = arrival_auc

    print(f"AUC para modelo de atraso na PARTIDA: {departure_auc:.4f}")
    print(f"AUC para modelo de atraso na CHEGADA: {arrival_auc:.4f}")

    # ===== MODELOS DE REGRESSÃO =====

    # Modelo para prever DEPARTURE_DELAY (quanto tempo vai atrasar)
    rf_departure = RandomForestRegressor(
        featuresCol=feature_col,
        labelCol="DEPARTURE_DELAY",
        numTrees=50,
        maxDepth=10
    )

    # Modelo para prever ARRIVAL_DELAY (quanto tempo vai atrasar)
    rf_arrival = RandomForestRegressor(
        featuresCol=feature_col,
        labelCol="ARRIVAL_DELAY",
        numTrees=50,
        maxDepth=10
    )

    # Treinar modelos de regressão
    print("Treinando modelos de regressão...")
    rf_departure_model = rf_departure.fit(train_df)
    rf_arrival_model = rf_arrival.fit(train_df)

    # Avaliar modelos de regressão
    departure_reg_predictions = rf_departure_model.transform(test_df)
    arrival_reg_predictions = rf_arrival_model.transform(test_df)

    evaluator_reg = RegressionEvaluator(metricName="rmse")

    # Configurar avaliador corretamente para Spark
    evaluator_reg.setLabelCol("DEPARTURE_DELAY")
    departure_rmse = evaluator_reg.evaluate(departure_reg_predictions)

    evaluator_reg.setLabelCol("ARRIVAL_DELAY")
    arrival_rmse = evaluator_reg.evaluate(arrival_reg_predictions)

    results["departure_regression_rmse"] = departure_rmse
    results["arrival_regression_rmse"] = arrival_rmse

    print(f"RMSE para modelo de tempo de atraso na PARTIDA: {departure_rmse:.2f} minutos")
    print(f"RMSE para modelo de tempo de atraso na CHEGADA: {arrival_rmse:.2f} minutos")

    # Combinar as previsões para análise - usando o API do Spark corretamente
    departure_predictions_selected = departure_reg_predictions.select(
        "FLIGHT_NUMBER",
        "DEPARTURE_DELAY",
        col("prediction").alias("PREDICTED_DEPARTURE_DELAY")
    )

    arrival_predictions_selected = arrival_reg_predictions.select(
        "FLIGHT_NUMBER",
        "ARRIVAL_DELAY",
        col("prediction").alias("PREDICTED_ARRIVAL_DELAY")
    )

    combined_predictions = departure_predictions_selected.join(
        arrival_predictions_selected,
        on="FLIGHT_NUMBER",
        how="inner"
    )

    # Calcular a correlação entre atrasos reais e previstos - usando a API do Spark
    departure_arrival_corr = combined_predictions.stat.corr("DEPARTURE_DELAY", "ARRIVAL_DELAY")
    predicted_departure_arrival_corr = combined_predictions.stat.corr(
        "PREDICTED_DEPARTURE_DELAY", "PREDICTED_ARRIVAL_DELAY"
    )

    results["departure_arrival_correlation"] = departure_arrival_corr
    results["predicted_departure_arrival_correlation"] = predicted_departure_arrival_corr

    print(f"Correlação entre atrasos reais (partida vs. chegada): {departure_arrival_corr:.4f}")
    print(f"Correlação entre atrasos previstos (partida vs. chegada): {predicted_departure_arrival_corr:.4f}")

    # Criar dataframe com previsões para análise adicional
    final_predictions = combined_predictions.withColumn(
        "DEPARTURE_VS_ARRIVAL_DIFF",
        col("ARRIVAL_DELAY") - col("DEPARTURE_DELAY")
    ).withColumn(
        "PREDICTED_DEPARTURE_VS_ARRIVAL_DIFF",
        col("PREDICTED_ARRIVAL_DELAY") - col("PREDICTED_DEPARTURE_DELAY")
    )

    # Mostrar estatísticas da diferença entre atrasos de partida e chegada
    print("Análise da diferença entre atrasos de partida e chegada:")
    final_predictions.select("DEPARTURE_VS_ARRIVAL_DIFF", "PREDICTED_DEPARTURE_VS_ARRIVAL_DIFF").summary().show()

    return results, final_predictions, rf_departure_model, rf_arrival_model

# 4. Treinar e avaliar os modelos
results, predictions_df, departure_model, arrival_model = train_and_evaluate_models(train_data, test_data)

# 5. Analisar relação entre atrasos de partida e chegada
print("\nAnálise de relação entre atrasos de partida e chegada:")

# Verificar probabilidade condicional: se atrasar na partida, qual a probabilidade de atrasar na chegada
delayed_departures = predictions_df.filter(col("DEPARTURE_DELAY") > threshold_minutes)
delayed_and_arrivals = delayed_departures.filter(col("ARRIVAL_DELAY") > threshold_minutes)

total_delayed_departures = delayed_departures.count()
total_delayed_both = delayed_and_arrivals.count()

if total_delayed_departures > 0:
    conditional_prob = total_delayed_both / total_delayed_departures
    print(f"Probabilidade de atraso na chegada dado atraso na partida: {conditional_prob:.2%}")
else:
    print("Não há voos com atraso na partida no conjunto de teste.")

# Verificar se atraso na partida geralmente se propaga ou é compensado
# Usando abordagem compatível com o Spark
propagation_df = predictions_df.withColumn(
    "delay_pattern",
    when(
        (col("DEPARTURE_DELAY") > threshold_minutes) & (col("ARRIVAL_DELAY") > threshold_minutes),
        lit("PROPAGATED")
    ).when(
        (col("DEPARTURE_DELAY") > threshold_minutes) & (col("ARRIVAL_DELAY") <= threshold_minutes),
        lit("COMPENSATED")
    ).when(
        (col("DEPARTURE_DELAY") <= threshold_minutes) & (col("ARRIVAL_DELAY") > threshold_minutes),
        lit("NEW_DELAY")
    ).otherwise(lit("ON_TIME"))
)

propagation_stats = propagation_df.groupBy("delay_pattern").count()

print("Padrões de propagação de atrasos:")
propagation_stats.show()

def predict_delays(new_data):
    # Verificar se o DataFrame já passou pelo pipeline de feature engineering
    if "scaled_features" in new_data.columns:
        # Os dados já estão processados, não precisamos aplicar o feature_model novamente
        processed_data = new_data
    else:
        # Aplicar o pipeline de feature engineering
        processed_data = feature_model.transform(new_data)

    # Fazer previsões
    departure_predictions = departure_model.transform(processed_data)
    arrival_predictions = arrival_model.transform(processed_data)

    # Combinar resultados
    departure_selected = departure_predictions.select(
        "FLIGHT_NUMBER",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL",
        col("prediction").alias("PREDICTED_DEPARTURE_DELAY")
    )

    arrival_selected = arrival_predictions.select(
        "FLIGHT_NUMBER",
        col("prediction").alias("PREDICTED_ARRIVAL_DELAY")
    )

    combined_results = departure_selected.join(
        arrival_selected,
        on="FLIGHT_NUMBER",
        how="inner"
    ).withColumn(
        "LIKELY_DEPARTURE_DELAY",
        when(col("PREDICTED_DEPARTURE_DELAY") > threshold_minutes, lit("SIM")).otherwise(lit("NÃO"))
    ).withColumn(
        "LIKELY_ARRIVAL_DELAY",
        when(col("PREDICTED_ARRIVAL_DELAY") > threshold_minutes, lit("SIM")).otherwise(lit("NÃO"))
    )

    return combined_results

# 7. Exemplo de uso para fazer previsões em novos dados
print("\nExemplo de previsões para novos voos:")
sample_flights = test_data.limit(5)
predicted_flights = predict_delays(sample_flights)

predicted_flights.select(
    "FLIGHT_NUMBER",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "PREDICTED_DEPARTURE_DELAY",
    "LIKELY_DEPARTURE_DELAY",
    "PREDICTED_ARRIVAL_DELAY",
    "LIKELY_ARRIVAL_DELAY"
).show()

# 8. Salvar modelos para uso futuro
departure_model.save("models/departure_delay_model")
arrival_model.save("models/arrival_delay_model")

print("Modelos salvos com sucesso.")