import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('eco_transport_data.csv')

mode_categories = ['Walk', 'Cycle', 'Scooter', 'Bus', 'Metro', 'Car', 'EV_Scooter', 'EV_Car']

df['mode'] = pd.Categorical(df['mode'], categories=mode_categories)

X = df[['distance_km', 'users_per_day']]
y = df['mode'].cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

y_pred_names = pd.Series(y_pred).map(lambda x: mode_categories[x]).values

plt.figure(figsize=(10, 6))
sns.barplot(x='mode', y='co2_g_per_km', data=df, estimator='mean', hue='mode', palette='Greens_r', legend=False)
plt.title("Average CO₂ Emissions per km by Transport Mode")
plt.ylabel("CO₂ Emissions (g/km)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
avg_dist = df.groupby('mode', observed=False)['distance_km'].mean().reset_index()
sns.lineplot(data=avg_dist, x='mode', y='distance_km', marker='o', color='teal')
plt.title("Average Distance Travelled by Transport Mode")
plt.ylabel("Average Distance (km)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='distance_km', y='total_co2_emission', hue='mode', s=100)
plt.title("Distance vs Total CO₂ Emission by Transport Mode")
plt.xlabel("Distance (km)")
plt.ylabel("Total CO₂ Emission (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

mode_user_sum = df.groupby('mode', observed=False)['users_per_day'].sum()
plt.figure(figsize=(8, 8))
plt.pie(mode_user_sum, labels=mode_user_sum.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("User Distribution by Transport Mode (Total Users per Day)")
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='mode', y='distance_km', data=df, hue='mode', palette='Set2', legend=False)
plt.title("Distance Distribution by Transport Mode")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
