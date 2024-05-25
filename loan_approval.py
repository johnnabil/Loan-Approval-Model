import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('loan_approval_dataset.csv')
df.head()

df.info()

df.columns = df.columns.str.strip()

data_clean = df.copy()

label_encoder = LabelEncoder()

data_clean['loan_status'] = label_encoder.fit_transform(data_clean['loan_status'])
data_clean['education'] = label_encoder.fit_transform(data_clean['education'])
data_clean['self_employed'] = label_encoder.fit_transform(data_clean['self_employed'])
data_clean = data_clean.drop(['loan_id'], axis=1)
data_clean.drop_duplicates()

data_clean.head()

corr_matrix = data_clean.corr()

sns.heatmap(corr_matrix, annot = True, cmap='coolwarm').figure.set_size_inches(12, 10)
plt.show()

corr_threshold = 0.9

for _ in data_clean.columns.to_list():

    corr_matrix[corr_matrix == 1] = 0

    to_drop = None

    correlated_features = [column for column in corr_matrix.columns if any(corr_matrix[column] > corr_threshold)]

    if len(correlated_features) > 0:
        to_drop = correlated_features[0]
    else:
        break

    print('The feature {} is highly correlated with another feature. Dropping it.\n'.format(to_drop))

    data_clean = data_clean.drop(to_drop, axis=1)

    corr_matrix = data_clean.corr()

sns.heatmap(corr_matrix, annot = True, cmap='coolwarm').figure.set_size_inches(12, 10)
plt.show()

scaler = StandardScaler()

scaled_data = data_clean.drop(columns=['loan_status'])
label = data_clean['loan_status']

numerical_columns = ['no_of_dependents', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                      'bank_asset_value']

scaled_data[numerical_columns] = scaler.fit_transform(scaled_data[numerical_columns])

print("Scaled Feature Variables (x):")
print(scaled_data.head())

print("\nTarget Variable (y):")
print(label.head())

cov_mat = np.cov(scaled_data.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\neigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]

plt.bar(range(1,11), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.show()


pca = PCA(n_components=4)
x_pca = pca.fit_transform(scaled_data)
pc_columns = [f'PC{i}' for i in range(1, 5)]
pc_df = pd.DataFrame(data=x_pca, columns=pc_columns)
pc_df['loan_status'] = label
pc_df.head()


for i, component in enumerate(pca.components_[:4], start=1):
    print(f"PC{i}:")
    features = scaled_data.columns
    loadings = zip(features, component)
    sorted_loadings = sorted(loadings, key=lambda x: abs(x[1]), reverse=True)
    for feature, loading in sorted_loadings:
        print(f"{feature}: {loading}")
    print()

pcs = [col for col in pc_df.columns if col.startswith('PC')]

for i in range(len(pcs)):
    for j in range(i+1, len(pcs)):
        sns.scatterplot(x=pcs[i], y=pcs[j], hue='loan_status', data=pc_df)
        plt.title(f'Scatter plot of {pcs[i]} vs {pcs[j]}')
        plt.show()

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    # kmeans.fit(data_clean.drop("loan_status", axis=1))
    kmeans.fit(pc_df.drop("loan_status", axis=1))
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, 'bx-')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


avg_silhouette = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    # kmeans.fit(data_clean.drop("loan_status", axis=1))
    kmeans.fit(pc_df.drop("loan_status", axis=1))
    # kmeans.fit(pc_df.drop("loan_status", axis=1))
    cluster_labels = kmeans.labels_
    # avg_silhouette.append(silhouette_score(data_clean.drop("loan_status", axis=1), cluster_labels))
    avg_silhouette.append(silhouette_score(pc_df.drop("loan_status", axis=1), cluster_labels))

plt.plot(range(2, 11),avg_silhouette, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in range(2, 6):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    # visualizer.fit(data_clean.drop('loan_status', axis=1))
    visualizer.fit(pc_df.drop('loan_status', axis=1))

km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
# y = km.fit_predict(data_clean.drop("loan_status", axis=1))
y = km.fit_predict(pc_df.drop("loan_status", axis=1))
# y = km.fit_predict(pc_df_tf.drop("loan_status", axis=1))
clusters = km.labels_
# score = silhouette_score(data_clean.drop("loan_status", axis=1), km.labels_)
score = silhouette_score(pc_df.drop("loan_status", axis=1), km.labels_)
print('Silhouetter Score: %.3f' % score)

visualizer = SilhouetteVisualizer(km, colors='yellowbrick') #, ax=ax[q-1][mod])
visualizer.fit(pc_df.drop('loan_status', axis=1))
visualizer.show()

plt.scatter(pc_df.iloc[clusters == 0, 0], pc_df.iloc[clusters == 0, 1], s = 30, c = 'red', label = 'Rejected')
plt.scatter(pc_df.iloc[clusters == 1, 0], pc_df.iloc[clusters == 1, 1], s = 30, c = 'green', label = 'Approved')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], s = 100, c = 'blue', label = 'Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.title('K-means Clustering with 2D PCA')


plt.legend()

plt.scatter(data_clean.loc[data_clean['loan_status'] == 0, 'cibil_score'], data_clean.loc[data_clean['loan_status'] == 0, 'loan_amount'], s=30, c='red', label='Rejected')
plt.scatter(data_clean.loc[data_clean['loan_status'] == 1, 'cibil_score'], data_clean.loc[data_clean['loan_status'] == 1, 'loan_amount'], s=30, c='green', label='Approved')

centroid_rejected = data_clean[data_clean['loan_status'] == 0][['cibil_score', 'loan_amount']].mean()
centroid_approved = data_clean[data_clean['loan_status'] == 1][['cibil_score', 'loan_amount']].mean()

plt.scatter(centroid_rejected['cibil_score'], centroid_rejected['loan_amount'], s=100, c='blue', label='Centroid - Rejected')
plt.scatter(centroid_approved['cibil_score'], centroid_approved['loan_amount'], s=100, c='blue', label='Centroid - Approved')

plt.legend()

db_index = round(davies_bouldin_score(pc_df.drop('loan_status', axis=1), y), 3)
db_index

k_values = range(2, 11)
db_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pc_df.drop('loan_status', axis=1))
    db_values.append(davies_bouldin_score(pc_df.drop('loan_status', axis=1), kmeans.labels_))

plt.plot(k_values, db_values)
plt.xlabel('k')
plt.ylabel('Davies-Bouldin index')
plt.title('Davies-Bouldin index for different k values')
plt.show()


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(data_clean.drop('loan_status', axis=1))


dendrogram(linkage(data_clean.drop('loan_status', axis=1), method="complete", metric="euclidean"))
plt.show()

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)

plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=1)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

pca_data = pc_df.drop('loan_status', axis=1)
kmedoids = KMedoids(n_clusters=2, init='k-medoids++', random_state=42)
kmedoids.fit(pca_data)
cluster_labels = kmedoids.labels_
pc_df['KMedoids_Cluster'] = cluster_labels

sns.scatterplot(x='PC1', y='PC2', hue='KMedoids_Cluster', data=pc_df)
plt.title('K-Medoids Clustering')
plt.show()

neighbors=NearestNeighbors(n_neighbors=2)
nbrs=neighbors.fit(pc_df.drop('loan_status', axis=1))
distances, indices=nbrs.kneighbors(pc_df.drop('loan_status', axis=1))
distances=np.sort(distances, axis = 0)
bbox=dict(boxstyle='round', pad=0.3, color='#FFDA47', alpha=0.6)
txt1=dict(textcoords='offset points', va='center', ha='center', fontfamily='serif', style='italic')
txt2=dict(textcoords='offset points', va='center', fontfamily='serif', style='italic')
kw=dict(arrowstyle='Simple, tail_width=0.1, head_width=0.4, head_length=1', color='black')
text_style=dict(fontweight='bold', fontfamily='serif')
fig=plt.figure(figsize=(14, 5))

distances_1=distances[:, 1]
ax1=fig.add_subplot(1, 3, (1, 2))
plt.plot(distances_1, color='#5829A7')
plt.xlabel('\nTotal', fontsize=9, **text_style)
plt.ylabel('Oldpeak\n', fontsize=9, **text_style)
for spine in ax1.spines.values():
  spine.set_color('None')
plt.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
plt.grid(axis='x', alpha=0)
plt.tick_params(labelsize=7)

plt.suptitle('DBSCAN Epsilon Value\n', fontsize=14, **text_style)
plt.show();

X = scaled_data
Y = data_clean['loan_status']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=42)
X_train.head()

X_PCA = pc_df.drop(columns=['loan_status'])
Y_PCA = pc_df['loan_status']
X_PCA_train,X_PCA_test,Y_PCA_train,Y_PCA_test = train_test_split(X_PCA,Y_PCA,test_size=0.3,stratify=Y_PCA,random_state=42)
X_PCA_train.head()

log_model = LogisticRegression()

pipeline = log_model.fit(X_train, Y_train)

predictions = log_model.predict(X_test)

predictions

conf_mat = confusion_matrix(Y_test, predictions)

conf_frame = pd.DataFrame(conf_mat,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

lr_accuracy = accuracy_score(Y_test, predictions)

print('Accuracy:', lr_accuracy*100,'%')

lr_precision = precision_score(Y_test, predictions, average='weighted')

print('Precision:', lr_precision*100, '%')

lr_recall = recall_score(Y_test, predictions, average='weighted')

print('Recall:', lr_recall*100, '%')

lr_f1 = f1_score(Y_test, predictions, average='weighted')

print('F1 score:', lr_f1*100, '%')

from sklearn import metrics
probs = log_model.predict_proba(X_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(Y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

plt.plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Logistic ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

rf_model.fit(X_train, Y_train)

predictions = rf_model.predict(X_test)

predictions

conf_mat = confusion_matrix(Y_test, predictions)

conf_frame = pd.DataFrame(conf_mat,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

rf_accuracy = accuracy_score(Y_test, predictions)

print('Accuracy:', rf_accuracy*100, '%')

rf_precision = precision_score(Y_test, predictions, average='weighted')

print('Precision:', rf_precision*100, '%')

rf_recall = recall_score(Y_test, predictions, average='weighted')

print('Recall:', rf_recall*100, '%')

rf_f1 = f1_score(Y_test, predictions, average='weighted')

print('F1 score:', rf_f1*100, '%')

probs = rf_model.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(Y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

plt.plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Random Forest ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

nb_model = GaussianNB()

nb_model.fit(X_train, Y_train)

predictions = nb_model.predict(X_test)

predictions

conf_mat = confusion_matrix(Y_test, predictions)

conf_frame = pd.DataFrame(conf_mat,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

nb_accuracy = accuracy_score(Y_test, predictions)

print('Accuracy:', nb_accuracy*100, '%')

nb_precision = precision_score(Y_test, predictions, average='weighted')

print('Precision:', nb_precision*100, '%')

nb_recall = recall_score(Y_test, predictions, average='weighted')

print('Recall:', nb_recall*100, '%')

nb_f1 = f1_score(Y_test, predictions, average='weighted')

print('F1 score:', nb_f1*100, '%')

probs = nb_model.predict_proba(X_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(Y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

plt.plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Gaussian ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

log_model = LogisticRegression()

log_model.fit(X_PCA_train, Y_PCA_train)

predictions_PCA = log_model.predict(X_PCA_test)

predictions_PCA

conf_mat_PCA = confusion_matrix(Y_PCA_test, predictions_PCA)

conf_frame_PCA = pd.DataFrame(conf_mat_PCA,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame_PCA, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

lr_accuracy_PCA = accuracy_score(Y_PCA_test, predictions_PCA)

print('Accuracy:', lr_accuracy_PCA*100,'%')

lr_precision_PCA = precision_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Precision:', lr_precision_PCA*100, '%')

lr_recall_PCA = recall_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Recall:', lr_recall_PCA*100, '%')

lr_f1_PCA = f1_score(Y_PCA_test, predictions_PCA, average='weighted')

print('F1 score:', lr_f1_PCA*100, '%')

probs = log_model.predict_proba(X_PCA_test)
preds = probs[:,1]
fprlog_pca, tprlog_pca, thresholdlog = metrics.roc_curve(Y_PCA_test, preds)
roc_auclog = metrics.auc(fprlog_pca, tprlog_pca)

plt.plot(fprlog_pca, tprlog_pca, 'b', label = 'AUC = %0.2f' % roc_auclog)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Logistic ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

rf_model.fit(X_PCA_train, Y_PCA_train)

predictions_PCA = rf_model.predict(X_PCA_test)

predictions_PCA

conf_mat = confusion_matrix(Y_PCA_test, predictions_PCA)

conf_frame = pd.DataFrame(conf_mat,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

rf_accuracy_PCA = accuracy_score(Y_PCA_test, predictions_PCA)

print('Accuracy:', rf_accuracy_PCA*100, '%')

rf_precision_PCA = precision_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Precision:', rf_precision_PCA*100, '%')

rf_recall_PCA = recall_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Recall:', rf_recall_PCA*100, '%')

rf_f1_PCA = f1_score(Y_PCA_test, predictions_PCA, average='weighted')

print('F1 score:', rf_f1_PCA*100, '%')

probs = rf_model.predict_proba(X_PCA_test)
preds = probs[:,1]
fprrfc_pca, tprrfc_pca, thresholdrfc = metrics.roc_curve(Y_PCA_test, preds)
roc_aucrfc = metrics.auc(fprrfc_pca, tprrfc_pca)

plt.plot(fprrfc_pca, tprrfc_pca, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Random Forest ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

nb_model = GaussianNB()

nb_model.fit(X_PCA_train, Y_PCA_train)

predictions_PCA = nb_model.predict(X_PCA_test)

predictions_PCA

conf_mat = confusion_matrix(Y_PCA_test, predictions_PCA)

conf_frame = pd.DataFrame(conf_mat,
                        columns=['Predicted Didn\'t Get Loan', 'Predicted Got Loan'],
                        index=['Actual Didn\'t Get Loan', 'Actual Got Loan'])

sns.heatmap(conf_frame, annot=True, fmt='d', annot_kws={"size": 16})
plt.show()

nb_accuracy_PCA = accuracy_score(Y_PCA_test, predictions_PCA)

print('Accuracy:', nb_accuracy_PCA*100, '%')

nb_precision_PCA = precision_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Precision:', nb_precision_PCA*100, '%')

nb_recall_PCA = recall_score(Y_PCA_test, predictions_PCA, average='weighted')

print('Recall:', nb_recall_PCA*100, '%')

nb_f1_PCA = f1_score(Y_PCA_test, predictions_PCA, average='weighted')

print('F1 score:', nb_f1_PCA*100, '%')

probs = nb_model.predict_proba(X_PCA_test)
preds = probs[:,1]
fprgau_pca, tprgau_pca, thresholdgau = metrics.roc_curve(Y_PCA_test, preds)
roc_aucgau = metrics.auc(fprgau_pca, tprgau_pca)

plt.plot(fprgau_pca, tprgau_pca, 'b', label = 'AUC = %0.2f' % roc_aucgau)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Gaussian ',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

metrics_df = pd.DataFrame({ 'Model' : ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes'],
                            'Metric' : ['Accuracy', 'Accuracy', 'Accuracy', 'Precision', 'Precision', 'Precision', 'Recall', 'Recall', 'Recall', 'F1 Score', 'F1 Score', 'F1 Score'],
                            'Score' : [lr_accuracy, rf_accuracy, nb_accuracy, lr_precision, rf_precision, nb_precision, lr_recall, rf_recall, nb_recall, lr_f1, rf_f1, nb_f1]})

sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df)
plt.legend(loc='lower right')
plt.show()

plt.plot(fprgau, tprgau, 'b', label = 'Gaussian', color='black')
plt.plot(fprrfc, tprrfc, 'b', label = 'Random Forest', color='green')
plt.plot(fprlog, tprlog, 'b', label = 'Logistic', color='grey')
plt.title('ROC',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})

metrics_df = pd.DataFrame({ 'Model' : ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Naive Bayes'],
                            'Metric' : ['Accuracy', 'Accuracy', 'Accuracy', 'Precision', 'Precision', 'Precision', 'Recall', 'Recall', 'Recall', 'F1 Score', 'F1 Score', 'F1 Score'],
                            'Score' : [lr_accuracy_PCA, rf_accuracy_PCA, nb_accuracy_PCA, lr_precision_PCA, rf_precision_PCA, nb_precision_PCA, lr_recall_PCA, rf_recall_PCA, nb_recall_PCA, lr_f1_PCA, rf_f1_PCA, nb_f1_PCA]})

sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df)
plt.legend(loc='lower right')
plt.show()

plt.plot(fprgau_pca, tprgau_pca, 'b', label = 'Gaussian', color='black')
plt.plot(fprrfc_pca, tprrfc_pca, 'b', label = 'Random Forest', color='green')
plt.plot(fprlog_pca, tprlog_pca, 'b', label = 'Logistic', color='grey')
plt.title('ROC',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 16})