import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

# ── 0. PAGE CONFIG & DESIGN LANGUAGE ──────────────────────────────────
# Page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="H1B Visa Analysis Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for modern UI with Dark Mode support
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* --- CSS Variables for Theming (Light Mode Default) --- */
    :root {
        --primary-color: #2563EB;
        --primary-hover-color: #1D4ED8;
        --text-color-strong: #111827;
        --text-color-normal: #374151;
        --text-color-light: #6B7280;
        --bg-color-main: #F9FAFB;
        --bg-color-card: #FFFFFF;
        --border-color: #E5E7EB;
        --font-family: 'Inter', sans-serif;
    }

    /* --- Dark Mode CSS Variables --- */
    [data-theme="dark"] {
        --primary-color: #3B82F6;
        --primary-hover-color: #60A5FA;
        --text-color-strong: #F9FAFB;
        --text-color-normal: #D1D5DB;
        --text-color-light: #9CA3AF;
        --bg-color-main: #111827;
        --bg-color-card: #1F2937;
        --border-color: #374151;
    }

    /* --- General Body & Font Styling --- */
    body, .stApp {
        font-family: var(--font-family);
        background-color: var(--bg-color-main);
        color: var(--text-color-normal);
    }

    /* --- Headers --- */
    h1, h2, h3 {
        font-family: var(--font-family);
        color: var(--text-color-strong);
    }
    h1 { font-weight: 700; font-size: 2.25rem; padding-bottom: 0.5rem; }
    h2 { font-weight: 600; font-size: 1.75rem; margin-top: 2rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border-color); }
    h3 { font-weight: 600; font-size: 1.25rem; margin-top: 1.5rem; }

    /* --- Card Styling for Plots and Metrics --- */
    .stPlotlyChart, .stMetric, .stDataFrame, .stRadio {
        background-color: var(--bg-color-card);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
        transition: box-shadow 0.3s ease-in-out, background-color 0.3s ease;
    }
    .stPlotlyChart:hover, .stMetric:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
    }
    .stRadio { padding-bottom: 14px; } /* Adjust padding for radio buttons */

    /* --- Sidebar Styling --- */
    .css-1d391kg {
        background-color: var(--bg-color-card);
        border-right: 1px solid var(--border-color);
    }
    .css-1d391kg h2 { border-bottom: none; }

    /* --- Button and Interactive Widget Styling --- */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--primary-hover-color);
        color: white;
    }
    .stButton>button:focus {
        box-shadow: 0 0 0 3px var(--primary-color) !important;
        outline: none;
    }

    /* --- Custom Navigation Bar --- */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: var(--bg-color-card);
        padding: 10px 40px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-color);
    }
    .navbar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    .navbar-date {
        font-size: 0.9rem;
        color: var(--text-color-light);
    }
    .main .block-container {
        padding-top: 6rem; 
    }
</style>
""", unsafe_allow_html=True)

# ── 1. HELPER FUNCTIONS & NAVIGATION ────────────────────────────────────

def create_navigation_bar():
    """Creates the top fixed navigation bar using HTML."""
    today = datetime.now().strftime("%B %d, %Y")
    st.markdown(f"""
        <div class="navbar">
            <div class="navbar-brand">📊 H1B Visa Dashboard</div>
            <div class="navbar-date">{today}</div>
        </div>
    """, unsafe_allow_html=True)

def apply_enhanced_style(fig, title_text=""):
    """Applies a consistent, modern style to Plotly figures, theme-aware."""
    current_theme = st.get_option("theme.base")
    
    if current_theme == "dark":
        bg_color = "#1F2937"
        font_color = "#F9FAFB"
        grid_color = "#374151"
    else:
        bg_color = "#FFFFFF"
        font_color = "#111827"
        grid_color = "#E5E7EB"

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color=font_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title=dict(
            text=title_text,
            font=dict(size=20, color=font_color, family="Inter, sans-serif"),
            x=0.5, xanchor="center"
        ),
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0)" if current_theme == "dark" else "rgba(255, 255, 255, 0.8)",
            bordercolor=grid_color, borderwidth=1,
            yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
        margin=dict(t=60, l=10, r=10, b=10),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    return fig

# ── 2. DATA LOADING (WITH CACHING) ───────────────────────────────────

@st.cache_data
def load_data():
    """Loads all necessary data files and performs initial processing."""
    # This function is unchanged as it deals with data, not presentation
    world = pd.read_csv("Data/worldcities.csv", encoding="utf-8")
    world["city_key"] = world["city_ascii"].str.strip().str.lower()
    world["iso3"] = world["iso3"].str.strip().str.upper()
    world_lite = world[["city_key","iso3","lat","lng"]].drop_duplicates()
    try:
        medium_df = pd.read_csv("Data/medium_companies.csv", encoding="latin1")
        large_df = pd.read_csv("Data/large_companies.csv", encoding="utf-8")
        medium_companies = set(medium_df['name'].str.strip().str.lower())
        large_companies = set(large_df['name'].str.strip().str.lower())
    except Exception as e:
        st.warning(f"Could not load company size data: {e}")
        medium_companies, large_companies = set(), set()
    data_dir = "Data"
    files = ["Employer Information 15-19.csv", "Employer Information 20-23.csv", "Employer Information 24.csv"]
    parts = []
    for fn in files:
        try:
            df = pd.read_csv(os.path.join(data_dir, fn), encoding="utf-16", sep='\t', low_memory=False, on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            for col in ["Initial Approval", "Initial Denial", "Continuing Approval", "Continuing Denial"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["Total Applications"] = df[["Initial Approval", "Initial Denial", "Continuing Approval", "Continuing Denial"]].sum(axis=1)
            df["company_lower"] = df["Employer (Petitioner) Name"].str.strip().str.lower()
            df["Company Size"] = "Small"
            df.loc[df["company_lower"].isin(medium_companies), "Company Size"] = "Medium"
            df.loc[df["company_lower"].isin(large_companies), "Company Size"] = "Large"
            df = df.drop(columns=["company_lower"])
            parts.append(df)
        except Exception as e:
            st.error(f"Error reading {fn}: {e}")
    if not parts:
        st.error("No H-1B data was loaded!")
        st.stop()
    h1b_df = pd.concat(parts, ignore_index=True)
    h1b_df["city_key"] = h1b_df["Petitioner City"].str.strip().str.lower()
    h1b_df["iso3"] = "USA"
    merged = h1b_df.merge(world_lite, on=["city_key","iso3"], how="left")
    missing = merged["lat"].isna()
    if missing.any():
        fb = merged.loc[missing, ["city_key"]].merge(world_lite.drop_duplicates("city_key"), on="city_key", how="left")
        merged.loc[missing, ["lat","lng"]] = fb[["lat","lng"]].values
    merged = merged.dropna(subset=["lat","lng"])
    wage_df = pd.read_csv("Data/oe_industry_wages_clean.csv")
    wage_df["NAICS"] = wage_df["NAICS"].astype(str)
    merged["Industry (NAICS) Code"] = merged["Industry (NAICS) Code"].astype(str)
    def normalize_naics(code):
        match = re.search(r'^(\d+)', str(code))
        return match.group(1) if match else code
    wage_df["NAICS_norm"] = wage_df["NAICS"].apply(normalize_naics)
    merged["NAICS_norm"] = merged["Industry (NAICS) Code"].apply(normalize_naics)
    return merged, wage_df

df, wage_df = load_data()


# ── 3. SIDEBAR & FILTERS ──────────────────────────────────────────────

st.sidebar.markdown("## Filters & Options")
st.sidebar.divider()

years = sorted(df["Fiscal Year"].unique())
min_year, max_year = int(min(years)), int(max(years))
year_range = st.sidebar.slider(
    "Select Fiscal Year Range",
    min_value=min_year, max_value=max_year,
    value=(min_year, max_year),
    step=1
)

sel_years = list(range(year_range[0], year_range[1] + 1))
fdf = df[df["Fiscal Year"].isin(sel_years)]

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About this Dashboard"):
    st.info(
        """
        **Data Sources**:
        - H-1B Employer Data Hub (USCIS)
        - Bureau of Labor Statistics, OEWS 2023
        - Company Size Dataset (Kaggle)
        - World Cities Data (SimpleMaps)
        
        **Creator**: [Shawn Wang](https://github.com/ShouzhiWang)
        **Optimizer**: Gemini
        """,
        icon="📊"
    )

# ── 4. VISUALIZATION FUNCTIONS (MODULAR) ──────────────────────────────

def create_dashboard_overview(data, wage_data):
    """Creates the main overview dashboard with KPIs and mini-charts."""
    st.header("Dashboard Overview")
    total_apps = data['Total Applications'].sum()
    total_approved = data['Initial Approval'].sum() + data['Continuing Approval'].sum()
    approval_rate = total_approved / total_apps if total_apps > 0 else 0
    top_state = data.groupby('Petitioner State')['Total Applications'].sum().idxmax()
    top_industry_code = data['Industry (NAICS) Code'].mode()[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Applications", value=f"{total_apps:,.0f}")
    with col2:
        st.metric(label="Overall Approval Rate", value=f"{approval_rate:.1%}")
    with col3:
        st.metric(label="Top State", value=top_state)
    with col4:
        st.metric(label="Top Industry (NAICS)", value=top_industry_code, help=f"The most frequent NAICS code is {top_industry_code}")
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Application Trends by Fiscal Year")
        year_stats = data.groupby("Fiscal Year")[['Initial Approval', 'Continuing Approval']].sum().reset_index()
        year_stats['Total'] = year_stats['Initial Approval'] + year_stats['Continuing Approval']
        fig = px.area(year_stats, x='Fiscal Year', y='Total', title="", height=300)
        fig.update_traces(line_color='#2563EB', fillcolor='rgba(37, 99, 235, 0.2)')
        fig = apply_enhanced_style(fig, "Total Approved Applications Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Distribution by Company Size")
        size_dist = data.groupby('Company Size')['Total Applications'].sum().reset_index()
        fig = px.pie(size_dist, names='Company Size', values='Total Applications', hole=0.4,
                     color_discrete_map={'Small':'#6366F1','Medium':'#F59E0B','Large':'#10B981'}, height=300)
        fig = apply_enhanced_style(fig, "Applications by Company Size")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_state_map(data, wage_data):
    """Creates the enhanced state choropleth map with metric toggles."""
    st.header("Geographic Analysis")
    st.subheader("H1B Applications by State")
    st.markdown("""
    This map shows the distribution of H1B applications. Toggle between "Application Count" and "Average Salary."
    Darker areas indicate higher values, helping to identify major H1B hubs.
    """)
    metric_choice = st.radio(
        "Select metric to display on map:",
        options=["Application Count", "Average Salary"],
        horizontal=True, key="map_metric"
    )
    state_stats = data.groupby("Petitioner State").agg({'Total Applications': 'sum'}).reset_index()
    z_col, color_scale, hover_template, title = "Total Applications", "Blues", "<b>%{location}</b><br>Total Applications: %{z:,}<extra></extra>", "Total H1B Applications by State"
    if metric_choice == "Average Salary":
        counts = data.groupby(["Petitioner State", "NAICS_norm"])["Total Applications"].sum().reset_index(name="h1b_count")
        merged = counts.merge(wage_data[["NAICS_norm", "Annual_Median"]], on="NAICS_norm", how="left").dropna(subset=["Annual_Median"])
        merged["h1b_count"] = pd.to_numeric(merged["h1b_count"], errors="coerce")
        merged["Annual_Median"] = pd.to_numeric(merged["Annual_Median"], errors="coerce")
        state_salary = merged.assign(weighted=merged["h1b_count"] * merged["Annual_Median"]).groupby("Petitioner State").agg(total_apps=("h1b_count", "sum"), sum_weighted=("weighted", "sum")).reset_index()
        state_salary["avg_salary"] = state_salary["sum_weighted"] / state_salary["total_apps"]
        state_stats = state_stats.merge(state_salary[["Petitioner State", "avg_salary"]], on="Petitioner State", how="left")
        z_col, color_scale, hover_template, title = "avg_salary", "Cividis", "<b>%{location}</b><br>Average Salary: $%{z:,.0f}<extra></extra>", "Average H1B Salary by State"
    fig = go.Figure(go.Choropleth(
        locations=state_stats["Petitioner State"],
        z=state_stats[z_col].fillna(0), locationmode="USA-states",
        colorscale=color_scale, marker_line_color="rgba(255,255,255,0.2)",
        hovertemplate=hover_template
    ))
    fig.update_layout(geo=dict(scope='usa', projection=dict(type='albers usa'), showlakes=False, bgcolor='rgba(0,0,0,0)'), height=600)
    fig = apply_enhanced_style(fig, title)
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_time_series(data):
    """Creates an enhanced time series chart."""
    st.header("Time Series Analysis")
    st.subheader("Application Types and Approval Rate Over Time")
    st.markdown("Annual trend of new vs. continuing applications, with the overall approval rate overlaid. A moving average smooths the trend.")
    rate = data.groupby("Fiscal Year").agg({
        "Initial Approval": "sum", "Initial Denial": "sum",
        "Continuing Approval": "sum", "Continuing Denial": "sum"
    }).reset_index()
    rate["total"] = rate[["Initial Approval", "Initial Denial", "Continuing Approval", "Continuing Denial"]].sum(axis=1)
    rate["approved"] = rate["Initial Approval"] + rate["Continuing Approval"]
    rate["approval_rate"] = rate["approved"] / rate["total"]
    rate['MA_3yr'] = rate['approval_rate'].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rate['Fiscal Year'], y=rate['Initial Approval'] + rate['Initial Denial'], name='New Applications', marker_color='#6366F1'))
    fig.add_trace(go.Bar(x=rate['Fiscal Year'], y=rate['Continuing Approval'] + rate['Continuing Denial'], name='Continuing Applications', marker_color='#A5B4FC'))
    fig.add_trace(go.Scatter(x=rate['Fiscal Year'], y=rate['approval_rate'], name='Approval Rate', mode='lines+markers', yaxis='y2', line=dict(color='#F59E0B', width=3)))
    fig.add_trace(go.Scatter(x=rate['Fiscal Year'], y=rate['MA_3yr'], name='3-Year Moving Average', mode='lines', yaxis='y2', line=dict(color='#F59E0B', width=2, dash='dash')))
    fig.update_layout(barmode='stack', yaxis=dict(title='Number of Applications'), 
                      yaxis2=dict(title='Approval Rate', overlaying='y', side='right', tickformat=".0%"),
                      hovermode='x unified')
    fig = apply_enhanced_style(fig, "H1B Application Trends & Approval Rate (Fiscal Year)")
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_wage_distribution(data, wage_data):
    """Creates a rich wage distribution violin plot."""
    st.header("Industry & Wage Analysis")
    st.subheader("Wage Distribution and H1B Count by Industry")
    st.markdown("Compare salary distributions across industries. Each 'violin' shows the salary range. Optionally overlay H1B application counts.")
    naics_map = wage_data.set_index("NAICS_norm")["NAICS_TITLE"].to_dict()
    available_naics = data["NAICS_norm"].dropna().unique().tolist()
    default_naics = [n for n in ['54', '51', '62'] if n in available_naics]
    sel_ind = st.multiselect(
        "Select industries to compare:", options=available_naics,
        format_func=lambda x: f"{x} - {naics_map.get(x, 'Unknown Industry')}",
        default=default_naics
    )
    show_counts = st.checkbox("Overlay H1B application count bar chart", value=True)
    if not sel_ind:
        st.warning("Please select at least one industry.")
        return
    wsub = wage_data[wage_data["NAICS_norm"].isin(sel_ind)]
    counts = data[data["NAICS_norm"].isin(sel_ind)].groupby("NAICS_norm")["Total Applications"].sum().reindex(sel_ind).fillna(0).to_frame("count")
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, naics in enumerate(sel_ind):
        row = wsub[wsub["NAICS_norm"] == naics]
        if row.empty: continue
        vals = row[["Annual_Pct10","Annual_Pct25","Annual_Median","Annual_Pct75","Annual_Pct90"]].values.flatten()
        display_name = naics_map.get(naics, naics)[:40]
        fig.add_trace(go.Violin(
            y=vals, name=display_name, box_visible=True, meanline_visible=True,
            fillcolor=colors[i % len(colors)], line_color='rgba(0,0,0,0.6)', opacity=0.7
        ))
    if show_counts:
        fig.add_trace(go.Bar(
            x=[naics_map.get(n, n)[:40] for n in sel_ind], y=counts["count"],
            name="H1B Applications", yaxis="y2", opacity=0.5, marker_color='#9CA3AF'
        ))
        fig.update_layout(yaxis2=dict(title="H1B Application Count", overlaying="y", side="right", showgrid=False))
    fig.update_layout(
        violingap=0.2, showlegend=False, height=600,
        yaxis=dict(title="Annual Wage ($)", tickprefix="$", tickformat=","),
        xaxis=dict(title="Industry")
    )
    fig = apply_enhanced_style(fig, "Industry Wage Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ── 5. MAIN APP LAYOUT ───────────────────────────────────────────────

create_navigation_bar()
st.title("H1B Visa Analysis Dashboard")
st.markdown(f"**Analysis Years: {year_range[0]} - {year_range[1]}** | Welcome to this interactive tool for exploring H1B visa trends.")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌍 Geographic Analysis", "📈 Time Series", "💼 Industry & Wage"])
with tab1:
    create_dashboard_overview(fdf, wage_df)
with tab2:
    create_enhanced_state_map(fdf, wage_df)
    st.subheader("Top 10 Cities by Application Count")
    st.markdown("Identify regional clusters of H1B sponsorship. This ranking reveals the metropolitan areas most active in attracting foreign talent.")
    city_stats = fdf.groupby("Petitioner City")['Total Applications'].sum().nlargest(10).reset_index()
    fig = px.bar(city_stats.sort_values('Total Applications'), x='Total Applications', y='Petitioner City', orientation='h', height=450)
    fig = apply_enhanced_style(fig, "Top 10 Cities by Application Volume")
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    create_enhanced_time_series(fdf)
    st.subheader("Approval Rate by Company Size")
    st.markdown("Compare the H1B application success rates for companies of different sizes. This can help in understanding if company size impacts application outcomes.")
    size_rate = fdf.groupby(["Fiscal Year", "Company Size"]).agg(
        approved=("Initial Approval", "sum"),
        total=("Total Applications", "sum")
    ).reset_index()
    size_rate['approved'] += fdf.groupby(["Fiscal Year", "Company Size"])['Continuing Approval'].sum().values
    size_rate["approval_rate"] = size_rate["approved"] / size_rate["total"]
    fig = px.line(
        size_rate, x="Fiscal Year", y="approval_rate", color="Company Size",
        markers=True, height=450,
        color_discrete_map={'Small':'#6366F1','Medium':'#F59E0B','Large':'#10B981'}
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode='x unified')
    fig = apply_enhanced_style(fig, "Approval Rate Trend by Company Size")
    st.plotly_chart(fig, use_container_width=True)
with tab4:
    create_enhanced_wage_distribution(fdf, wage_df)
    st.subheader("Top 10 Companies by Application Count")
    st.markdown("Discover the leading H1B sponsors. This ranking shows the employers who file the most petitions.")
    comp_stats = fdf.groupby("Employer (Petitioner) Name")['Total Applications'].sum().nlargest(10).reset_index()
    fig = px.bar(comp_stats.sort_values('Total Applications'), x='Total Applications', y='Employer (Petitioner) Name', orientation='h', height=450)
    fig = apply_enhanced_style(fig, "Top 10 Companies by Application Volume")
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: var(--text-color-light); font-size: 0.9rem; padding-top: 1rem;'>H1B Visa Analysis Dashboard | Built with Streamlit and Plotly</div>", unsafe_allow_html=True)
