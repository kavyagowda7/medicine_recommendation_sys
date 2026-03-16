import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="💊 Medicine Recommendation System",
    layout="wide",
)

# --- 1. Load Data ---
# Use Streamlit's cache to load these large files only once
@st.cache_data 
def load_all_data():
    """Loads all necessary data and recommendation components."""
    try:
        # Load the main dataset
        df = pd.read_csv('medicine.csv')
        # Standardize/clean column names for UI display
        df.columns = ['Index', 'Drug Name', 'Reason', 'Description']
        
        # Load mappings (medicine_dict.pkl)
        with open('medicine_dict.pkl', 'rb') as f:
             medicine_dict = pickle.load(f)
        
        # Load the similarity matrix (similarity.pkl)
        with open('similarity.pkl', 'rb') as f:
             similarity_matrix = pickle.load(f)

        # Prepare necessary data structures from medicine_dict
        # The 'Drug_Name' key in the dict holds the index-to-name mapping
        index_to_drug_name = medicine_dict['Drug_Name'] 
        
        # Create a reverse map: Drug Name to Index for quick lookups
        # We assume the index starts from 0 (the dictionary keys are 0-based integers)
        drug_name_to_index = {name: i for i, name in index_to_drug_name.items()}

        return df, similarity_matrix, drug_name_to_index, index_to_drug_name
        
    except FileNotFoundError as e:
        st.error(f"File Error: {e}. Please ensure all three files ('medicine.csv', 'medicine_dict.pkl', 'similarity.pkl') are in the same directory.")
        return pd.DataFrame(), None, {}, {}
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame(), None, {}, {}

# Load the data
df, similarity, drug_to_idx, idx_to_drug = load_all_data()

# --- 2. Recommendation Logic ---
def recommend_medicines(drug_name, data_df, sim_matrix, d_to_i, i_to_d, num_recommendations=5):
    """
    Core function to fetch recommendations using the similarity matrix.
    """
    if drug_name not in d_to_i:
        return ["Drug name not found in the index map."]
        
    # Get the index of the medicine in the matrix
    idx = d_to_i[drug_name]
    
    # Get the similarity scores for that medicine
    # .flatten() converts the array from [[...]] to [...]
    sim_scores = list(enumerate(sim_matrix[idx])) 
    
    # Sort the medicines based on the similarity score
    # x[1] is the similarity score (the second element of the tuple)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar medicines (excluding itself)
    # The first element [0] is the drug itself, so skip it by starting at index 1
    top_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    
    # Map the indices back to drug names
    recommendations = [i_to_d[index] for index in top_indices]
    
    return recommendations


# --- 3. Streamlit App Layout ---
st.title("💊 Medicine & Drug Recommendation System")
st.markdown("Select a medicine to view its details and receive recommendations for similar treatments.")

if not df.empty and similarity is not None and drug_to_idx:
    
    # Get the list of all drug names for the dropdown
    all_drugs = df['Drug Name'].sort_values().unique()

    # --- Sidebar for Selection ---
    with st.sidebar:
        st.header("Select Medicine")
        selected_drug = st.selectbox(
            "Choose a Drug Name:",
            all_drugs
        )
        st.markdown("---")
        st.info(f"Loaded **{len(df)}** unique medicine records.")

    # --- Main Content Area ---
    if selected_drug:
        # Get the row corresponding to the selected drug
        # Note: We use .iloc[0] because the drug names might not be unique in the original CSV index, 
        # but they must be unique in the DataFrame index we use for display.
        selected_data = df[df['Drug Name'] == selected_drug].iloc[0]

        # 1. Display Selected Medicine Information
        st.header(f"Details for: **{selected_drug}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primary Reason")
            st.metric(label="Reason for Use", value=selected_data['Reason'])

        with col2:
            st.subheader("Description")
            # Use markdown for multiline display
            st.markdown(f"**Description:** {selected_data['Description']}")

        st.markdown("---")
        
        # 2. Recommendation Section
        st.subheader("Top Recommended Alternatives")

        # Add a spinner while the recommendation is being calculated
        with st.spinner(f'Finding top 5 similar alternatives for {selected_drug}...'):
            time.sleep(0.5) 
            recommendations = recommend_medicines(selected_drug, df, similarity, drug_to_idx, idx_to_drug, num_recommendations=5)
        
        if recommendations:
            cols = st.columns(5) # Display recommendations in 5 columns
            for i, rec_drug in enumerate(recommendations):
                cols[i].success(f"**{i+1}.** {rec_drug}")
        else:
            st.warning("No recommendations found or an error occurred during lookup.")

else:
    st.error("Application could not fully load data. Please check the file names and structure in the console logs.")

# --- End of Streamlit Script ---