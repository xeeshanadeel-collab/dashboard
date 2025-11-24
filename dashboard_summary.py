import streamlit as st
import pandas as pd
import plotly.express as px
import io
import os
import random # Used for simulation

# Set Streamlit page configuration
st.set_page_config(
    page_title="Loan Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AI Commentary Function (SIMULATED) ---
def generate_ai_commentary(df_filtered):
    """
    Simulates AI commentary generation based on key filtered metrics.
    In a real application, this would call the Gemini API.
    """
    if df_filtered.empty:
        return "No data selected. Please adjust your filters to generate commentary."
    
    total_accounts = len(df_filtered)
    total_os = df_filtered.get('OS30112024', pd.Series(dtype=float)).sum()
    
    # 1. Classification Focus (e.g., finding the riskiest status)
    if 'Classification' in df_filtered.columns:
        classification_os = df_filtered.groupby('Classification')['OS30112024'].sum().sort_values(ascending=False)
        top_classification = classification_os.index[0]
        top_os_amount = classification_os.iloc[0]
        
        # 2. Segment Focus
        if 'Business Segment' in df_filtered.columns:
            segment_os = df_filtered.groupby('Business Segment')['OS30112024'].sum().sort_values(ascending=False)
            top_segment = segment_os.index[0]
            top_segment_share = top_os_amount / total_os * 100 if total_os > 0 else 0
            
            commentary_parts = [
                f"The current filtered portfolio consists of **{total_accounts:,} accounts** with a total outstanding of **{total_os:,.2f}**.",
                f"The largest concentration of risk, in terms of outstanding amount, is currently found in the **{top_classification}** classification, representing **{top_os_amount:,.2f}**.",
                f"This risk is predominantly driven by the **{top_segment}** segment, which contributes approximately **{top_segment_share:.1f}%** of the total outstanding amount."
            ]
        else:
            commentary_parts = [
                f"The current filtered portfolio consists of **{total_accounts:,} accounts** with a total outstanding of **{total_os:,.2f}**.",
                f"The largest concentration of risk, in terms of outstanding amount, is currently found in the **{top_classification}** classification, representing **{top_os_amount:,.2f}**."
            ]

    # --- SIMULATED AI TONALITY ---
    # In a real app, you would use a prompt like: 
    # "Analyze the following JSON data for loan portfolio performance. Highlight the top 3 segments of risk and growth. [JSON data]"
    
    simulated_summary = " ".join(commentary_parts)
    
    return simulated_summary

# Use caching to load and preprocess data only once
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data(uploaded_file, sheet_name=None):
    """
    Loads data from the uploaded file (CSV or XLSX) and preprocesses it.
    If XLSX is provided, sheet_name specifies which sheet to load.
    """
    if uploaded_file is None:
        return pd.DataFrame(), None # Return empty DataFrame and None sheets
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    df = pd.DataFrame()
    all_sheets = None

    try:
        # Check if we are reading from the beginning
        uploaded_file.seek(0)
        
        if file_extension == '.csv':
            # Read CSV file
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            # Read all sheets in Excel file to get sheet names
            xls = pd.ExcelFile(uploaded_file)
            all_sheets = xls.sheet_names
            
            # Load the specified sheet, defaulting to the first sheet if none is provided
            sheet_to_load = sheet_name if sheet_name and sheet_name in all_sheets else all_sheets[0]
            uploaded_file.seek(0) # Rewind for the parse operation if needed
            df = xls.parse(sheet_to_load)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return pd.DataFrame(), None
            
        # --- Preprocessing Steps ---
        
        # Drop rows where all elements are NaN (if any)
        df.dropna(how='all', inplace=True)

        # Convert key columns to numeric, coercing errors to NaN
        numeric_cols = ['OS30112024', 'Credit Limit']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors='coerce')

        # Drop rows where the key numerical columns are NaN after conversion
        # Ensure the columns exist before trying to drop NaNs
        cols_to_check = [col for col in numeric_cols if col in df.columns]
        if cols_to_check:
             df.dropna(subset=cols_to_check, inplace=True)

        # Clean up string columns by filling NaNs with 'Unknown' and converting to string
        string_cols = ['Business Segment', 'Classification', 'District', 'RM']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
            
        # Add a unique identifier for counting accounts
        df['Account_ID'] = df.index

        return df, all_sheets

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return pd.DataFrame(), None

# --- Sidebar: Data Upload & Selection (Remains in sidebar) ---
st.sidebar.title("Data Upload & Selection")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Loan Data (CSV or XLSX)",
    type=['csv', 'xlsx', 'xls']
)

sheet_name = None
df = pd.DataFrame()
all_sheets = None

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    uploaded_file.seek(0) # Rewind for initial read

    if file_extension in ['.xlsx', '.xls']:
        try:
            xls = pd.ExcelFile(uploaded_file)
            all_sheets = xls.sheet_names
            uploaded_file.seek(0) # Rewind after reading sheet names
            
            # Sheet selection widget
            sheet_name = st.sidebar.selectbox(
                "Select Sheet for Dashboard",
                options=all_sheets
            )
        except Exception as e:
            st.sidebar.error(f"Could not read sheets from Excel file: {e}")
            uploaded_file = None 

    # Load the actual data using the determined sheet (or None for CSV)
    if uploaded_file:
        df, _ = load_data(uploaded_file, sheet_name=sheet_name)
    

# --- Main Dashboard Logic ---
if df.empty:
    st.title("Loan Portfolio Performance Dashboard")
    st.info("Please upload a CSV or XLSX file using the sidebar to begin dashboard analysis.")
    if uploaded_file and not df.empty:
        st.warning("Data loaded, but essential columns ('OS30112024' or 'Credit Limit') might be missing or corrupted.")
else:
    st.title("Loan Portfolio Performance Dashboard")
    
    # --- AI Commentary Section ---
    st.markdown("---")
    st.subheader("Portfolio Commentary")
    
    # Placeholder for commentary - updated after filter application
    ai_commentary_placeholder = st.container()
    
    st.markdown("---")
    st.markdown("Use the filters below to refine the data analysis.")

    # --- Filters (Moved to Main Body inside an Expander) ---
    with st.expander("Filter Criteria", expanded=True):
        
        # Use columns to lay out the filters horizontally
        col_seg, col_class, col_dist, col_rm = st.columns(4)

        # Initialize defaults for safe masking
        selected_segments = []
        selected_classifications = []
        selected_districts = []
        selected_rms = []
        os_range = (df['OS30112024'].min(), df['OS30112024'].max()) # Default full range

        # 1. Multi-select filter for Business Segment
        if 'Business Segment' in df.columns:
            all_segments = df['Business Segment'].unique()
            selected_segments = col_seg.multiselect(
                "Business Segment",
                options=all_segments,
                default=all_segments
            )
        else:
            col_seg.warning("Missing 'Business Segment'.")

        # 2. Multi-select filter for Classification
        if 'Classification' in df.columns:
            all_classifications = df['Classification'].unique()
            selected_classifications = col_class.multiselect(
                "Classification Status",
                options=all_classifications,
                default=all_classifications
            )
        else:
            col_class.warning("Missing 'Classification'.")

        # 3. Multi-select filter for District
        if 'District' in df.columns:
            all_districts = df['District'].unique()
            selected_districts = col_dist.multiselect(
                "District",
                options=all_districts,
                default=all_districts
            )
        else:
            col_dist.warning("Missing 'District'.")
        
        # 4. Multi-select filter for RM
        if 'RM' in df.columns:
            all_rms = df['RM'].unique()
            selected_rms = col_rm.multiselect(
                "Relationship Manager (RM)",
                options=all_rms,
                default=all_rms
            )
        else:
            col_rm.warning("Missing 'RM'.")
        
        st.markdown("---")
        

    # --- Apply Filters ---
    
    # Define a mask for all filters
    mask = pd.Series(True, index=df.index)

    # Apply filters only if the columns and selections exist
    if selected_segments and 'Business Segment' in df.columns:
        mask &= df['Business Segment'].isin(selected_segments)
    if selected_classifications and 'Classification' in df.columns:
        mask &= df['Classification'].isin(selected_classifications)
    if selected_districts and 'District' in df.columns:
        mask &= df['District'].isin(selected_districts)
    if selected_rms and 'RM' in df.columns:
        mask &= df['RM'].isin(selected_rms)
        
    # Apply OS range filter
    if 'OS30112024' in df.columns:
        mask &= (df['OS30112024'] >= os_range[0]) & (df['OS30112024'] <= os_range[1])

    df_filtered = df[mask]
    
    # --- Generate and Display AI Commentary ---
    with ai_commentary_placeholder:
        commentary = generate_ai_commentary(df_filtered)
        st.info(commentary)

    # --- Dashboard Summary ---
    st.markdown("---")
    st.subheader(f"Data Snapshot: {len(df_filtered)} Accounts Selected")

    # Show a warning if no data is selected
    if df_filtered.empty:
        st.warning("No data matches the current filter selection. Please adjust the filters.")
    else:
        # --- KPI Metrics ---
        total_os = df_filtered.get('OS30112024', pd.Series(dtype=float)).sum()
        total_credit_limit = df_filtered.get('Credit Limit', pd.Series(dtype=float)).sum()
        total_accounts = len(df_filtered)

        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Total Accounts",
            value=f"{total_accounts:,}"
        )
        col2.metric(
            label="Total Outstanding Amount (OS30112024)",
            value=f"{total_os:,.2f}"
        )
        col3.metric(
            label="Total Credit Limit",
            value=f"{total_credit_limit:,.2f}"
        )
        
        st.markdown("---")

        # --- Visualizations ---
        
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Outstanding Amount by Business Segment (Bar Chart)
        if 'Business Segment' in df_filtered.columns:
            segment_summary = df_filtered.groupby('Business Segment')['OS30112024'].sum().reset_index()
            fig_segment = px.bar(
                segment_summary,
                x='Business Segment',
                y='OS30112024',
                title='Outstanding Amount by Business Segment',
                color='Business Segment',
                template='plotly_white',
                labels={'OS30112024': 'Total Outstanding Amount'}
            )
            fig_segment.update_layout(showlegend=False)
            chart_col1.plotly_chart(fig_segment, use_container_width=True)
        else:
            chart_col1.info("Cannot display Business Segment chart: Column missing.")


        # Chart 2: Accounts Distribution by Classification (Pie Chart)
        if 'Classification' in df_filtered.columns:
            classification_counts = df_filtered.groupby('Classification')['Account_ID'].count().reset_index()
            classification_counts.columns = ['Classification', 'Count']
            fig_classification = px.pie(
                classification_counts,
                names='Classification',
                values='Count',
                title='Account Distribution by Classification Status',
                hole=0.3,
                template='plotly_white'
            )
            fig_classification.update_traces(textposition='inside', textinfo='percent+label')
            chart_col2.plotly_chart(fig_classification, use_container_width=True)
        else:
            chart_col2.info("Cannot display Classification chart: Column missing.")

        st.markdown("---")
        
        ## 📊 RM Performance Summary
        if all(col in df_filtered.columns for col in ['RM', 'OS30112024', 'Credit Limit']):
            rm_summary = df_filtered.groupby('RM').agg(
                Total_OS=('OS30112024', 'sum'),
                Total_Limit=('Credit Limit', 'sum'),
                Account_Count=('Account_ID', 'count')
            ).reset_index()
            
            # Calculate Utilization (OS / Limit)
            rm_summary['Utilization_Rate'] = (rm_summary['Total_OS'] / rm_summary['Total_Limit']).fillna(0).map('{:.2%}'.format)
            rm_summary['Total_OS'] = rm_summary['Total_OS'].map('{:,.2f}'.format)
            rm_summary['Total_Limit'] = rm_summary['Total_Limit'].map('{:,.2f}'.format)
            
            st.subheader("Relationship Manager (RM) Performance Summary")
            st.dataframe(rm_summary, use_container_width=True)
        else:
            st.info("RM Performance Summary skipped: Missing 'RM', 'OS30112024', or 'Credit Limit' column.")

        st.markdown("---")


        # --- Detailed Data Table ---
        st.subheader("Filtered Data Details")
        st.dataframe(df_filtered.head(100), use_container_width=True) # Display top 100 rows for performance

# Footer/Execution Note
st.markdown("###### Dashboard generated using Streamlit and Plotly.")