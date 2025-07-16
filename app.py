import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport

# Title
st.title("Smart Dataset Cleaning & Analysis Tool")

# File upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Original Dataset")
    st.dataframe(df.head())

    # Show initial report
    if st.button("Generate Initial Report"):
        st.write("Generating report... This may take a moment.")
        profile = ProfileReport(df, title="Initial Data Report", minimal=True)
        profile_html = profile.to_html()
        st.components.v1.html(profile_html, height=600, scrolling=True)

    # Missing value handling
    def handle_missing_values(df):
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(include='object').columns
        num_imputer = SimpleImputer(strategy='median')
        df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])
        return df_clean

    # Outlier treatment (IQR capping)
    def treat_outliers(df, factor=1.5):
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=np.number).columns
        for col in num_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        return df_clean

    if st.button("Clean Dataset"):
        st.write("Cleaning data...")
        df_no_missing = handle_missing_values(df)
        df_cleaned = treat_outliers(df_no_missing)

        st.write("### Cleaned Dataset")
        st.dataframe(df_cleaned.head())

        # Post-cleaning report
        st.write("Generating post-cleaning report...")
        profile_clean = ProfileReport(df_cleaned, title="Post-Cleaning Report", minimal=True)
        profile_html_clean = profile_clean.to_html()
        st.components.v1.html(profile_html_clean, height=600, scrolling=True)

        # Select column for comparison visualization
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_col = st.selectbox("Select column for before/after boxplot comparison", numeric_cols)

        if selected_col:
            fig, axs = plt.subplots(1,2, figsize=(12,5))
            sns.boxplot(y=df[selected_col], ax=axs[0])
            axs[0].set_title(f'Before Cleaning: {selected_col}')
            sns.boxplot(y=df_cleaned[selected_col], ax=axs[1])
            axs[1].set_title(f'After Cleaning: {selected_col}')
            st.pyplot(fig)

        # Option to download cleaned data
        csv = df_cleaned.to_csv(index=False).encode()
        st.download_button(label="Download Cleaned CSV", data=csv, file_name='cleaned_dataset.csv', mime='text/csv')
