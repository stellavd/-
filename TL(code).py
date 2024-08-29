import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Εμφάνιση τίτλου εφαρμογής 
st.title("•Εφαρμογή Ανάλυσης Δεδομένων•")

# Upload δεδομένων 
uploaded_file = st.file_uploader("Επιλογή αρχείου CSV ή Excel:", type=["csv", "xlsx"])

#Έλεγχος αν  ανέβηκε το αρχείο και τι τύπος είναι  
if uploaded_file is not None:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Πίνακας Δεδομένων :")
    st.dataframe(df)


    # Δημιουργία των tabs
    tab_info, tab1, tab2, tab3, tab4 = st.tabs(["Πληροφορίες Εφαρμογής", "2D Visualization", "EDA", "Classification", "Clustering"])

    
    #Tab οπτικοποίησης
    with tab1:
        st.header("2D Οπτικοποίηση Δεδομένων")

        # Εδώ επιλέγει αλγόριθμο
        algorithm = st.selectbox("Επιλέξτε αλγόριθμο μείωσης διάστασης", ["PCA", "t-SNE"])

        # Αυτόματη ανίχνευση στηλών του πίνακα 
        feature_columns = [col for col in df.columns if col != df.columns[-1]]
        output_column_name = df.columns[-1]

        if not feature_columns:
            st.error("Αδυναμία οπτικοποίησης, δεν βρέθηκαν απαραίτητα χαρακτηριστικά.")
        else:
            if algorithm == "PCA":
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(df[feature_columns])
                reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
            elif algorithm == "t-SNE":
                tsne = TSNE(n_components=2)
                reduced_data = tsne.fit_transform(df[feature_columns])
                reduced_df = pd.DataFrame(reduced_data, columns=['t-SNE1', 't-SNE2'])

            reduced_df[output_column_name] = df[output_column_name].values

            # Δημιουργία scatter plot
            fig, ax = plt.subplots()
            sns.scatterplot(data=reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1], hue=output_column_name, ax=ax)
            ax.set_title(f'Οπτικοποίηση με {algorithm}')
            st.pyplot(fig)

    # EDA Tab
    with tab2:
        st.header("Exploratory Data Analysis (EDA)")

        # Αυτόματη ανίχνευση και δημιουργία EDA 
        st.subheader("Κατανομή Χαρακτηριστικών")
        num_features_to_plot = min(3, len(df.columns) - 1)  #Να μην μπει το label
        feature_columns = [col for col in df.columns if col != df.columns[-1]]
        for feature in feature_columns[:num_features_to_plot]:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f'Κατανομή Χαρακτηριστικού {feature}')
            st.pyplot(fig)

        st.subheader("Διασπορά Χαρακτηριστικών")
        if len(feature_columns) > 1:
            fig, ax = plt.subplots()
            sns.pairplot(df[feature_columns + [df.columns[-1]]], hue=df.columns[-1])
            st.pyplot(fig)

    # Classification Tab
    with tab3:
        st.header("Κατηγοριοποίηση Δεδομένων")

        # Επιλέγει αλγόριθμο 
        classifier = st.selectbox("Επιλέξτε αλγόριθμο κατηγοριοποίησης", ["KNN", "SVM"])

        # Ορισμός παραμέτρου από χρίση 
        if classifier == "KNN":
            k = st.slider("Επιλέξτε το k για τον KNN", min_value=1, max_value=20, value=5)
        elif classifier == "SVM":
            kernel = st.selectbox("Επιλέξτε τον τύπο για τον SVM", ["linear", "poly", "rbf"])

        # Προετοιμασία των δεδομένων
        feature_columns = [col for col in df.columns if col != df.columns[-1]]
        X = df[feature_columns]
        y = df[df.columns[-1]]
        le = LabelEncoder()
        y = le.fit_transform(y) 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if classifier == "KNN":
            model = KNeighborsClassifier(n_neighbors=k)
        elif classifier == "SVM":
            model = SVC(kernel=kernel)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Εκτίμηση
        st.write("Αναφορά Κατηγοριοποίησης:")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Η ακρίβεια του μοντέλου είναι: {accuracy:.2f}")

    # Clustering Tab
    with tab4:
        st.header("Ομαδοποίηση Δεδομένων")

        # επιλέγει αλγόριθμο
        clustering_algorithm = st.selectbox("Επιλέξτε αλγόριθμο ομαδοποίησης", ["KMeans"])

        #Ορισμός παραμέτρου
        if clustering_algorithm == "KMeans":
            k_clusters = st.slider("Επιλέξτε τον αριθμό των ομάδων (k)", min_value=1, max_value=20, value=3)

        # Προετοιμασία των δεδομένων
        feature_columns = [col for col in df.columns if col != df.columns[-1]]
        X = df[feature_columns]

        if clustering_algorithm == "KMeans":
            model = KMeans(n_clusters=k_clusters)
            clusters = model.fit_predict(X)
            df['Cluster'] = clusters

            #scatter plot για το clustering
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=feature_columns[0], y=feature_columns[1], hue='Cluster', palette='viridis', ax=ax)
            ax.set_title(f'Ομαδοποίηση KMeans με {k_clusters} ομάδες')
            st.pyplot(fig)

            # Silhouette Score
            silhouette_avg = silhouette_score(X, clusters)
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # tab Αποτελέσματα και Σύγκριση
    with st.sidebar:
        st.header("Αποτελέσματα και Σύγκριση")

        # Σύγκριση Αποτελεσμάτων
        compare_button = st.button("Σύγκριση Αποτελεσμάτων")

        if compare_button:
            st.subheader("Αποτελέσματα Κατηγοριοποίησης")
            st.write(f"Κατηγοριοποίηση Ακρίβεια: {accuracy:.2f}")
            st.write("Κατηγοριοποίηση Αναφορά:")
            st.text(classification_report(y_test, y_pred, target_names=le.classes_))

            st.subheader("Αποτελέσματα Ομαδοποίησης")
            st.write(f"Silhouette Score για KMeans: {silhouette_avg:.2f}")

    # Tab για Πληροφορίες Εφαρμογής
    with tab_info:
        st.header("Πληροφορίες Εφαρμογής")
        st.write("""
        Η εφαρμογή αυτή εμπεριέχει εργαλεία για την ανάλυση δεδομένων μέσω της οπτικοποίησης τους, 
        εξερευνητικής ανάλυσης, κατηγοριοποίησης και ομαδοποίησης. Φορτώνοντας ένα αρχείο τύπου CSV ή Excel 
        και διαλέγοντας αλγόριθμο τις επιλογής σας μπορείτε να δείτε κατευθείαν τα αποτελέσματα βάση των επιλογών σας.
        """)
