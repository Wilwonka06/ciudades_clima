import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('datos_clima.csv')

#4 preparación de datos para clustering
#Aplicar K-Means con (K=3) grupos de clima
X = df[["temperatura_promedio_A", "precipitación_promedio_A"]]
kmeans = KMeans(n_clusters=3, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  


#modelar y predecir clusters
df['cluster'] = kmeans.fit_predict(X_scaled)

#Visualizar los clusters
plt.figure(figsize=(12, 6))
plt.scatter(df['temperatura_promedio_A'], df['precipitación_promedio_A'], c=df['cluster'], cmap='Set1', s=100)
plt.xlabel('Temperatura Promedio Anual')
plt.ylabel('Precipitación Promedio Anual')
plt.colorbar(label='Cluster')
plt.grid()

plt.title('Clustering de Clima con K-Means (K=3)')
plt.show()  



df.info()
print(df.head(20))

# Guardar el scaler y el modelo
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "modelo_clima.pkl")
