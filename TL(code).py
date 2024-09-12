import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Title of app 
st.title('•Εφαρμογή Ανάλυσης Δεδομένων•')

# Tab section
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📄 Δεδομένα", "📊 Οπτικοποίηση", "📈 EDA", "🔍 Επιλογή Χαρακτηριστικών", "📊 Κατηγοριοποίηση", "ℹ️ Πληροφορίες"])

# Upload data
uploaded_file = st.file_uploader("Επιλογή αρχείου CSV, TSV ή Excel:", type=["csv", "tsv", "xlsx"])

# Check if file is up
if uploaded_file is not None:
    try:
        # Check type of file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep='\t')
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')

        # Change non-numerical column σε κατηγοριοποιημένες τιμές
        for column in data.columns:
            if data[column].dtype == 'object':
                # if column is type 'object', make it category
                data[column] = pd.Categorical(data[column]).codes
        else:
            # if numerical do nothing με διαχείριση κενών τιμών
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # fill κενών τιμών with median
        data = data.fillna(data.mean())


        # fill κενών τιμών με median or 0
        data = data.fillna(data.mean())

        # check SxF structure
        if data.shape[1] < 2:
            st.error("Ο πίνακας πρέπει να έχει τουλάχιστον δύο στήλες: χαρακτηριστικά και ετικέτες.")
        else:
            with tab1:
                st.write("Πίνακας Δεδομένων:")
                st.write(data)

            with tab2:
                # standardize data before applying PCA/UMAP
                features = data.iloc[:, :-1]  #all columns, not last
                labels = data.iloc[:, -1]      # last column (labels)

                # convert categorical labels to numeric codes
                le = LabelEncoder()
                numeric_labels = le.fit_transform(labels)

                st.subheader("2D and 3D Visualizations")
                st.write("Οι παρακάτω οπτικοποιήσεις βασίζονται σε PCA και UMAP για μείωση διάστασης.")

                # scale data
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                # PCA 2D Plot
                pca = PCA(n_components=2)
                pca_2d = pca.fit_transform(scaled_features)
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=numeric_labels, cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
                ax.add_artist(legend1)
                plt.title("PCA 2D Visualization")
                st.pyplot(fig)

                # PCA 3D Plot
                pca_3d = PCA(n_components=3).fit_transform(scaled_features)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=numeric_labels, cmap='plasma')
                legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
                ax.add_artist(legend1)
                plt.title("PCA 3D Visualization")
                st.pyplot(fig)

                # UMAP 2D Plot
                umap_2d = UMAP(n_components=2).fit_transform(scaled_features)
                fig, ax = plt.subplots()
                scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=numeric_labels, cmap='coolwarm')
                legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
                ax.add_artist(legend1)
                plt.title("UMAP 2D Visualization")
                st.pyplot(fig)

                # UMAP 3D Plot
                umap_3d = UMAP(n_components=3).fit_transform(scaled_features)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(umap_3d[:, 0], umap_3d[:, 1], umap_3d[:, 2], c=numeric_labels, cmap='spring')
                legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
                ax.add_artist(legend1)
                plt.title("UMAP 3D Visualization")
                st.pyplot(fig)

            # Exploratory Data Analysis (EDA) tab
            with tab3:
                st.subheader("Exploratory Data Analysis (EDA)")
                
                # EDA Plot 1: Histogram of selected feature
                st.write("1. Κατανομή χαρακτηριστικών (Histogram)")
                selected_column = st.selectbox("Επιλέξτε χαρακτηριστικό για την κατανομή:", data.columns[:-1])
                fig, ax = plt.subplots()
                sns.histplot(data[selected_column], kde=True, ax=ax)
                plt.title(f"Histogram of {selected_column}")
                st.pyplot(fig)

                # EDA Plot 2: Correlation Heatmap
                st.write("2. Correlation Heatmap")
                if features.shape[1] > 1:
                    fig, ax = plt.subplots()
                    corr_matrix = pd.DataFrame(scaled_features, columns=features.columns).corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    plt.title("Correlation Heatmap")
                    st.pyplot(fig)
                else:
                    st.write("Δεν υπάρχουν αρκετά χαρακτηριστικά για τον υπολογισμό του πίνακα συσχέτισης.")

            # Feature Selection tab
            with tab4:
                st.subheader("Επιλογή Χαρακτηριστικών")
                num_features = st.number_input("Επιλέξτε αριθμό χαρακτηριστικών:", 
                                                min_value=1, 
                                                max_value=features.shape[1], 
                                                value=1)

                # Feature Selection with SelectKBest
                selector = SelectKBest(f_classif, k=num_features)
                X_new = selector.fit_transform(features, numeric_labels)
                selected_features = features.columns[selector.get_support()]

                st.write("Dataset με επιλεγμένα χαρακτηριστικά:")
                st.write(X_new)

                # Save selected features for later use
                st.session_state.selected_features = X_new
                st.session_state.labels = numeric_labels

            # Classification tab
            with tab5:
                st.subheader("Κατηγοριοποίηση")
                
                # User selects classification algorithms
                algorithm = st.selectbox("Επιλέξτε αλγόριθμο:", ["k-nearest neighbors (knn)"])
                if algorithm == "k-nearest neighbors (knn)":
                    k = st.number_input("Επιλέξτε την παράμετρο k (αριθμός γειτόνων):", min_value=1, value=3)

                if st.button("Εκτέλεση Κατηγοριοποίησης"):
                    # Check if features have been selected
                    if 'selected_features' not in st.session_state:
                        st.error("Πρέπει πρώτα να εκτελέσετε Επιλογή Χαρακτηριστικών.")
                    else:
                        # Prepare the original and new datasets
                        X_train_orig, X_test_orig, y_train, y_test = train_test_split(features, numeric_labels, test_size=0.3, random_state=42)
                        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(st.session_state.selected_features, numeric_labels, test_size=0.3, random_state=42)

                        # Train and evaluate on original dataset
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train_orig, y_train)

                        # Evaluate on original dataset
                        y_pred_orig = knn.predict(X_test_orig)
                        acc_orig = accuracy_score(y_test, y_pred_orig)
                        f1_orig = f1_score(y_test, y_pred_orig, average='weighted')

                        # Handle multiclass case in roc_auc_score
                        try:
                            roc_auc_orig = roc_auc_score(y_test, knn.predict_proba(X_test_orig), multi_class='ovr')
                        except ValueError:
                            roc_auc_orig = "Not applicable for this classification"

                        st.write("Αποτελέσματα για το αρχικό dataset:")
                        st.write(f"Accuracy: {acc_orig:.4f}")
                        st.write(f"F1-Score: {f1_orig:.4f}")
                        st.write(f"ROC AUC: {roc_auc_orig}")

                        # Train evaluate on feature-selected dataset
                        knn.fit(X_train_new, y_train_new)
                        y_pred_new = knn.predict(X_test_new)
                        acc_new = accuracy_score(y_test_new, y_pred_new)
                        f1_new = f1_score(y_test_new, y_pred_new, average='weighted')

                        # Handle multiclass case in roc_auc_score
                        try:
                            roc_auc_new = roc_auc_score(y_test_new, knn.predict_proba(X_test_new), multi_class='ovr')
                        except ValueError:
                            roc_auc_new = "Not applicable for this classification"

                        st.write("Αποτελέσματα για το dataset με μειωμένα χαρακτηριστικά:")
                        st.write(f"Accuracy: {acc_new:.4f}")
                        st.write(f"F1-Score: {f1_new:.4f}")
                        st.write(f"ROC AUC: {roc_auc_new}")

                        # compare results
                        st.subheader("Αποτελέσματα και Σύγκριση:")
                        st.write("Συγκρίνουμε τις μετρικές απόδοσης μεταξύ του αρχικού dataset και του dataset με μειωμένα χαρακτηριστικά.")

                        comparison_df = pd.DataFrame({
                            'Dataset': ['Αρχικό', 'Με Μειωμένα Χαρακτηριστικά'],
                            'Accuracy': [acc_orig, acc_new],
                            'F1-Score': [f1_orig, f1_new],
                            'ROC AUC': [roc_auc_orig, roc_auc_new]
                        })

                        st.write(comparison_df)

                        # Plot comparison
                        fig, ax = plt.subplots()
                        comparison_df.set_index('Dataset').plot(kind='bar', ax=ax)
                        plt.title('Σύγκριση Απόδοσης')
                        plt.xticks(rotation=0)
                        st.pyplot(fig)

                        # which algorithm better
                        st.write("Συμπέρασμα:")
                        best_dataset = comparison_df.iloc[comparison_df[['Accuracy', 'F1-Score', 'ROC AUC']].mean(axis=1).idxmax()]['Dataset']
                        st.write(f"Το {best_dataset} dataset παρουσιάζει την καλύτερη απόδοση με βάση τις μετρικές.")

        # Info Tab
        with tab6:
            st.subheader("App info:")
            st.write("""
                Η εφαρμογή αυτή εμπεριέχει εργαλεία για την ανάλυση δεδομένων 
                     μέσο της οπτικοποίησης τους σε 2D και 3D χρησιμοποιώντας PCA και UMAP, 
                     εξερευνητικής ανάλυσης(EDA), επιλογής των χαρακτηριστικών και κατηγοριοποίησης τους
                      και σύγκρισης των αποτελεσμάτων της απόδοσης των αλγορίθμων. 
                     Φορτώνοντας ένα αρχείο τύπου CSV, TSV ή Excel μπορείτε να δείτε κατευθείαν τα αποτελέσματα βάση των επιλογών σας.
            """)

            st.subheader("Ομάδα Ανάπτυξης:")
            st.write("""
                \n- **Αθανασιάδης Δημήτριος (inf2021011)**: Υλοποίηση Προδιαγραφών Πίνακα & Tabs Μηχανικής Μάθησης.
                \n- **Βασιλειάδου Στυλιανή (inf2021025)**:  Υλοποίηση των Visualization Tabs & Αποτελέσματα αλγορίθμων και Σύγκριση.
                \n- **Ξενίδης Δημήτριος (inf2021256)**: Υλοποίηση Φόρτωσης Δεδομένων & Info Tab.
            """)
    except Exception as e:
        st.error(f"Σφάλμα κατά την ανάγνωση του αρχείου: {e}")
else:
    st.write("Ανεβάστε ένα αρχείο CSV, TSV ή Excel")
