import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

# ── STYLING CONFIGURATIONS ──────────────────────────────────────────────
# Set page config with custom theme
st.set_page_config(
    page_title="H1B Visa Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
    <style>
    /* Main title styling */
    .main .block-container h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    
    /* Subheader styling */
    .main .block-container h2 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.8rem;
        font-weight: 500;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Subsubheader styling */
    .main .block-container h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.4rem;
        font-weight: 500;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
    
    /* Card styling */
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        background-color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric styling */
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ── 1) LOAD & GEOCODE WORLD CITIES ──────────────────────────────────────
@st.cache_data
def load_world_lite():
    world = pd.read_csv("Data/worldcities.csv", encoding="utf-8")
    world["city_key"] = world["city_ascii"].str.strip().str.lower()
    world["iso3"]     = world["iso3"].str.strip().str.upper()
    return world[["city_key","iso3","lat","lng"]].drop_duplicates()

world_lite = load_world_lite()

# ── 2) LOAD COMPANY SIZE LISTS ─────────────────────────────────────────
@st.cache_data
def load_company_sizes():
    try:
        # Load medium and large companies
        try:
            medium_df = pd.read_csv("Data/medium_companies.csv", encoding="utf-8")
        except:
            medium_df = pd.read_csv("Data/medium_companies.csv", encoding="latin1")
        large_df  = pd.read_csv("Data/large_companies.csv", encoding="utf-8")

        # Build lowercase name sets using the correct column 'name'
        medium_set = set(medium_df['name'].str.strip().str.lower())
        large_set  = set(large_df['name'].str.strip().str.lower())
        

        return medium_set, large_set

    except Exception as e:
        st.error(f"Error loading company size data: {e}")
        return set(), set()

medium_companies, large_companies = load_company_sizes()

# ── 3) LOAD H‑1B DATA ───────────────────────────────────────────────────
@st.cache_data
def load_h1b_data():
    data_dir = "Data"
    files = [
        "Employer Information 15-19.csv",
        "Employer Information 20-23.csv",
        "Employer Information 24.csv",
    ]
    parts = []
    for fn in files:
        path = os.path.join(data_dir, fn)
        try:
            df = pd.read_csv(
                path,
                encoding="utf-16",
                sep='\t',
                low_memory=False,
                on_bad_lines='skip'
            )
            df.columns = df.columns.str.strip()

            # numeric coercion
            for col in ["Initial Approval","Initial Denial",
                        "Continuing Approval","Continuing Denial"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df["Total Applications"] = (
                df["Initial Approval"]
                + df["Initial Denial"]
                + df["Continuing Approval"]
                + df["Continuing Denial"]
            )

            # normalize employer name
            df["company_lower"] = (
                df["Employer (Petitioner) Name"]
                .str.strip()
                .str.lower()
            )

            # categorize size (default = Small)
            df["Company Size"] = "Small"
            df.loc[df["company_lower"].isin(medium_companies), "Company Size"] = "Medium"
            df.loc[df["company_lower"].isin(large_companies),  "Company Size"] = "Large"

            # debug counts
            
            df = df.drop(columns=["company_lower"])
            parts.append(df)

        except Exception as e:
            st.error(f"Error reading {fn}: {e}")

    if not parts:
        st.error("No H-1B data loaded!")
        st.stop()

    return pd.concat(parts, ignore_index=True)

df = load_h1b_data()

# ── 4) MERGE WITH LAT/LNG ───────────────────────────────────────────────
df["city_key"] = df["Petitioner City"].str.strip().str.lower()
df["iso3"]     = "USA"
merged = df.merge(world_lite, on=["city_key","iso3"], how="left")

# fill missing geocodes
missing = merged["lat"].isna()
if missing.any():
    fb = (
        merged.loc[missing, ["city_key"]]
        .merge(world_lite.drop_duplicates("city_key"), on="city_key", how="left")
    )
    merged.loc[missing, ["lat","lng"]] = fb[["lat","lng"]].values

merged = merged.dropna(subset=["lat","lng"])

# ── 5) FILTER UI ────────────────────────────────────────────────────────
# Add a title to the sidebar
st.sidebar.markdown("## Filter Options")

years = sorted(merged["Fiscal Year"].unique())
min_year, max_year = min(years), max(years)
year_range = st.sidebar.slider(
    "Fiscal Years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
    format="%d"
)

with st.sidebar.expander("About H‑1B Application Types"):
    st.markdown("""
    **Initial Approval**: First-time approval for a worker in a specialty occupation.
    **Continuing Approval**: Extensions or amendments for workers already in H‑1B status.
    """)

# Add a divider in the sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "Data Source: [H-1B Employer Data Hub](https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub), [Bureau of Labor Statistics, Occupational Employment and Wage Statistics 2023](https://www.bls.gov/oes/), [Company Size Dataset](https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset), [World Cities Data](https://simplemaps.com/data/world-cities)"
    "\n\nNote: 'N/A' or 'NaN' indicates missing or unavailable data."
    "\n\nCreated by [Shawn Wang](https://github.com/ShouzhiWang)",
    icon=":material/info:"
)

# Convert year range to list of years for filtering
sel = list(range(year_range[0], year_range[1] + 1))
fdf = merged[merged["Fiscal Year"].isin(sel)]

# ── 6) VISUALS ──────────────────────────────────────────────────────────
# Main title
st.title("H1B Visa Analysis Dashboard")

# Add introduction paragraph
st.markdown("""
Welcome to the H1B Visa Analysis Dashboard, where you can explore U.S. petition volumes, approval rates, 
geographic hotspots, and salary context across industries. This interactive tool helps you understand 
H1B visa trends, employer patterns, and regional distributions.
""")

# Function to apply consistent styling to all plots
def apply_consistent_style(fig):
    fig.update_layout(
        font=dict(
            family="Helvetica Neue, sans-serif",
            size=12,
            color="#1F2937"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, l=10, r=10, b=10),
        title=dict(
            font=dict(
                size=20,
                color="#1E3A8A"
            ),
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        )
    )
    return fig

## Choropleth by State
st.subheader("H1B Applications by State")
st.markdown("""
This map shows total H1B petitions by state. Hover over a state to see detailed approval/denial breakdowns. 
Darker colors indicate higher petition volumes, helping identify major H1B sponsorship hubs.
""")
state_stats = (
    fdf.groupby("Petitioner State")
    .agg({'Total Applications': 'sum', 
          'Initial Approval':'sum',
          'Initial Denial':'sum',
          'Continuing Approval':'sum',
          'Continuing Denial':'sum'})
    .reset_index()
)
fig_map = go.Figure(data=go.Choropleth(
    locations=state_stats['Petitioner State'],
    z=state_stats['Total Applications'],
    locationmode='USA-states',
    colorscale='Viridis',
    marker_line_color='white',
    marker_line_width=0.5,
    text=state_stats.apply(
        lambda x: (
            f"State: {x['Petitioner State']}<br>"
            f"Total: {x['Total Applications']:,}<br>"
            f"Init App: {x['Initial Approval']:,}<br>"
            f"Init Den: {x['Initial Denial']:,}"
        ), axis=1
    ),
    hovertemplate="%{text}<extra></extra>"
))
fig_map.update_layout(
    geo=dict(scope='usa', projection=dict(type='albers usa'), showlakes=True),
    margin={"r":0,"t":30,"l":0,"b":0}, height=600,
    title="H1B Applications by State"
)
fig_map = apply_consistent_style(fig_map)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

## Top 10 Cities
st.subheader("Top 10 Cities by H1B Applications")
st.markdown("""
Top 10 cities ranked by petition count. Use this to identify regional clusters of H1B sponsorship. 
The data reveals which metropolitan areas are most active in hiring foreign talent.
""")
city_stats = (
    fdf.groupby(["Petitioner City", "Petitioner State"])
    .agg({
        'Total Applications': 'sum',
        'Initial Approval': 'sum',
        'Initial Denial': 'sum',
        'Continuing Approval': 'sum',
        'Continuing Denial': 'sum'
    })
    .reset_index()
)

# Sort by total applications and get top 10
top10_cities = city_stats.nlargest(10, 'Total Applications').iloc[::-1]

# Create the bar chart
fig_cities = px.bar(
    top10_cities,
    x='Total Applications',
    y='Petitioner City',
    orientation='h',
    title="Top 10 Cities by H1B Applications",
    height=450,
    hover_data=['Petitioner State', 'Initial Approval', 'Initial Denial', 'Continuing Approval', 'Continuing Denial']
)

# Update hover template to show detailed information
fig_cities.update_traces(
    hovertemplate="<b>%{y}, %{customdata[0]}</b><br>" +
                 "Total Applications: %{x:,}<br>" +
                 "Initial Approvals: %{customdata[1]:,}<br>" +
                 "Initial Denials: %{customdata[2]:,}<br>" +
                 "Continuing Approvals: %{customdata[3]:,}<br>" +
                 "Continuing Denials: %{customdata[4]:,}<extra></extra>"
)

fig_cities = apply_consistent_style(fig_cities)
st.plotly_chart(fig_cities, use_container_width=True)

# Create a summary table in an expander
with st.expander("View Detailed City Statistics"):
    city_table = top10_cities.copy()
    city_table["Approval Rate"] = (
        (city_table["Initial Approval"] + city_table["Continuing Approval"]) /
        city_table["Total Applications"]
    ).round(3)

    # Format numbers for display
    for col in ["Total Applications", "Initial Approval", "Initial Denial", 
                "Continuing Approval", "Continuing Denial"]:
        city_table[col] = city_table[col].map("{:,}".format)
    city_table["Approval Rate"] = city_table["Approval Rate"].map("{:.1%}".format)

    st.dataframe(
        city_table,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

## Applications by Year
st.subheader("Applications by Fiscal Year")
st.markdown("""
See how the mix of new vs. renewal H1B filings has evolved over time. This chart breaks down initial 
applications (first-time petitions) and continuing applications (extensions/amendments) by fiscal year.
""")
year_stats = fdf.groupby("Fiscal Year")[[
    "Initial Approval","Initial Denial","Continuing Approval","Continuing Denial"
]].sum().reset_index()
year_stats["Total Applications"] = year_stats.drop(columns="Fiscal Year").sum(axis=1)

fig_bar = px.bar(
    year_stats, x='Fiscal Year',
    y=['Continuing Approval','Continuing Denial','Initial Approval','Initial Denial'],
    title="Applications by Fiscal Year", height=450, barmode='stack'
)
fig_bar.update_traces(hovertemplate="%{fullData.name}: %{y:,}<extra></extra>")
max_total = year_stats["Total Applications"].max()
fig_bar.add_trace(go.Scatter(
    x=year_stats['Fiscal Year'],
    y=[max_total]*len(year_stats),
    mode='markers', marker_opacity=0,
    customdata=year_stats['Total Applications'],
    hovertemplate="Total Apps: %{customdata:,}<extra></extra>",
    showlegend=False
))
fig_bar.update_layout(hovermode='x unified')

# Apply consistent styling to all figures
fig_bar = apply_consistent_style(fig_bar)

st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

## Top 10 Industries
st.subheader("Top 10 Industries by NAICS Code")
st.markdown("""
Explore which industries are most active in H1B sponsorship. The NAICS codes reveal the economic sectors 
driving demand for foreign talent, from technology to healthcare to professional services.
""")
all_states = ["Nationally"] + sorted(fdf["Petitioner State"].dropna().unique().tolist())
sel_state = st.selectbox("Select State", options=all_states, index=0)
df_state = fdf if sel_state=="Nationally" else fdf[fdf["Petitioner State"]==sel_state]
naics_stats = df_state.groupby("Industry (NAICS) Code")['Total Applications'].sum().reset_index()
top10 = naics_stats.nlargest(10, 'Total Applications').iloc[::-1]
fig_naics = px.bar(
    top10, x='Total Applications', y='Industry (NAICS) Code',
    orientation='h', title=f"Top 10 NAICS - {sel_state}", height=450
)
fig_naics.update_traces(hovertemplate="Apps: %{x:,}<extra></extra>")

# Apply consistent styling to all figures
fig_naics = apply_consistent_style(fig_naics)

st.plotly_chart(fig_naics, use_container_width=True)

st.markdown("---")

## Top 10 Companies
st.subheader("Top 10 Companies")
st.markdown("""
Discover the leading H1B sponsors. This ranking shows which employers file the most petitions, 
highlighting the companies most invested in hiring foreign talent.
""")
comp_stats = df_state.groupby("Employer (Petitioner) Name")['Total Applications'].sum().reset_index()
top10c = comp_stats.nlargest(10, 'Total Applications').iloc[::-1]
fig_comp = px.bar(
    top10c, x='Total Applications', y='Employer (Petitioner) Name',
    orientation='h', title=f"Top 10 Companies - {sel_state}", height=450
)
fig_comp.update_traces(hovertemplate="Apps: %{x:,}<extra></extra>")

# Apply consistent styling to all figures
fig_comp = apply_consistent_style(fig_comp)

st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

## Company Size Distribution
with st.expander("Company Size Distribution (experimental, may not be accurate)"):
    st.subheader("H1B Applications by Company Size")
    st.markdown("""
    Analyze how company size correlates with H1B sponsorship patterns. This distribution shows whether 
    small, medium, or large firms dominate the H1B landscape.
    """)
    size_dist = df_state.groupby('Company Size')['Total Applications'].sum().reset_index()
    fig_pie = px.pie(
        size_dist, names='Company Size', values='Total Applications',
        title=f"Distribution by Company Size - {sel_state}",
        color_discrete_map={'Small':'#636EFA','Medium':'#EF553B','Large':'#00CC96'}
    )
    fig_pie.update_traces(
        textposition='inside', textinfo='percent+label',
        hovertemplate="%{label}: %{value:,} apps (%{percent:.1%})<extra></extra>"
    )

    # Apply consistent styling to all figures
    fig_pie = apply_consistent_style(fig_pie)

    st.plotly_chart(fig_pie, use_container_width=True)
    st.write(size_dist.sort_values('Total Applications', ascending=False))



st.subheader("Overall H‑1B Approval Rate Over Time")
rate = (
    fdf
    .groupby("Fiscal Year")
    .agg({
        "Initial Approval": "sum",
        "Initial Denial": "sum",
        "Continuing Approval": "sum",
        "Continuing Denial": "sum"
    })
    .reset_index()
)

rate["total"] = (
    rate["Initial Approval"] + 
    rate["Initial Denial"] + 
    rate["Continuing Approval"] + 
    rate["Continuing Denial"]
)

rate["approval_rate"] = (
    (rate["Initial Approval"] + rate["Continuing Approval"])
    / rate["total"]
)

fig_rate = px.line(
    rate,
    x="Fiscal Year",
    y="approval_rate",
    title="H‑1B Approval Rate Over Time",
    labels={"approval_rate":"Approval Rate"}
)
fig_rate.update_yaxes(tickformat=".0%")
fig_rate.update_traces(mode="lines+markers", hovertemplate="%{y:.1%}")

# Apply consistent styling to all figures
fig_rate = apply_consistent_style(fig_rate)

st.plotly_chart(fig_rate, use_container_width=True)






st.markdown("---")

# ── COMBINED APPROVAL RATE BY STATE OR INDUSTRY ───────────────────────────
st.subheader("H‑1B Approval Rate Over Time")

# ── GROUPING DIMENSION: RADIO ────────────────────────────────────────────
group_dim = st.radio(
    "Compare approval rates by:",
    options=["Petitioner State", "Industry (NAICS) Code"],
    index=1  # default to Industry when you want industries in a state
)

# Add checkboxes for visualization options
col1, col2 = st.columns(2)
with col1:
    show_all = st.checkbox("Show all items (may be slow)", value=False)
with col2:
    show_average = st.checkbox("Show average across all items", value=False)

# ── PICK TOP 3 FOR MULTI‐SELECT ─────────────────────────────────────────
totals = (
    fdf
    .groupby(group_dim)["Total Applications"]
    .sum()
    .sort_values(ascending=False)
)

if show_all:
    sel_items = sorted(totals.index)
else:
    top3 = totals.head(3).index.tolist()
    sel_items = st.multiselect(
        f"Select {group_dim}(s):",
        options=sorted(totals.index),
        default=top3
    )
    if not sel_items:
        sel_items = top3

# ── STATE FILTER (only show when grouping by Industry) ───────────────────
if group_dim == "Industry (NAICS) Code":
    all_states = ["All"] + sorted(fdf["Petitioner State"].dropna().unique())
    sel_state = st.selectbox(
        "Select State to filter by:",
        options=all_states,
        index=all_states.index("All")
    )
    
    # apply state filter
    if sel_state != "All":
        df_plot = fdf[fdf["Petitioner State"] == sel_state]
    else:
        df_plot = fdf.copy()
else:
    df_plot = fdf  # full national data for state comparison

# ── FILTER TO SELECTED GROUPS ───────────────────────────────────────────
df_plot = df_plot[df_plot[group_dim].isin(sel_items)]

# ── AGGREGATE & COMPUTE RATES ───────────────────────────────────────────
rate_df = (
    df_plot
    .groupby(["Fiscal Year", group_dim])
    .agg(
        init_app=("Initial Approval",   "sum"),
        cont_app=("Continuing Approval","sum"),
        init_den=("Initial Denial",     "sum"),
        cont_den=("Continuing Denial",  "sum"),
    )
    .reset_index()
)
rate_df["total"]    = rate_df[["init_app","init_den","cont_app","cont_den"]].sum(axis=1)
rate_df["approved"] = rate_df["init_app"] + rate_df["cont_app"]
rate_df["approval_rate"] = rate_df["approved"] / rate_df["total"]

# Calculate average if requested
if show_average:
    avg_rate = (
        rate_df
        .groupby("Fiscal Year")
        .agg(
            total=("total", "sum"),
            approved=("approved", "sum")
        )
        .reset_index()
    )
    avg_rate["approval_rate"] = avg_rate["approved"] / avg_rate["total"]
    avg_rate[group_dim] = "Average"

# ── PLOT ────────────────────────────────────────────────────────────────
st.subheader(f"H‑1B Approval Rate Over Time by {group_dim}"
             + (f" in {sel_state}" if group_dim=="Industry (NAICS) Code" and sel_state!="All" else ""))

# Combine regular data with average if requested
plot_df = pd.concat([rate_df, avg_rate]) if show_average else rate_df

fig = px.line(
    plot_df,
    x="Fiscal Year",
    y="approval_rate",
    color=group_dim,
    markers=True,
    labels={"approval_rate":"Approval Rate"},
    title="Approval Rate Over Time"
)
fig.update_yaxes(tickformat=".0%")
fig.update_traces(hovertemplate="%{y:.1%}")

# Make average line more prominent and adjust other lines if average is shown
if show_average:
    # Make average line prominent
    fig.update_traces(
        line=dict(width=4, dash='dash'),
        selector=dict(name="Average")
    )
    
    # Reduce opacity of other lines and make them thinner
    for trace in fig.data:
        if trace.name != "Average":
            trace.line.width = 1
            trace.line.color = trace.line.color.replace('rgb', 'rgba').replace(')', ', 0.3)')
            trace.marker.size = 4
            trace.marker.opacity = 0.3

# Add hover mode for better interaction
fig.update_layout(
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

# Apply consistent styling to all figures
fig = apply_consistent_style(fig)

st.plotly_chart(fig, use_container_width=True)


# ── Load OEWS wage data ─────────────────────────────────────────────────
wage_df = pd.read_csv("Data/oe_industry_wages_clean.csv")
# Ensure NAICS is same type as your fdf
wage_df["NAICS"] = wage_df["NAICS"].astype(str)
fdf["Industry (NAICS) Code"] = fdf["Industry (NAICS) Code"].astype(str)

# Normalize NAICS codes by extracting just the numeric part
def normalize_naics(code):
    # Extract first number from formats like "54" or "54 - Professional, Scientific, and Technical Services"
    match = re.search(r'^(\d+)', str(code))
    return match.group(1) if match else code

wage_df["NAICS_norm"] = wage_df["NAICS"].apply(normalize_naics)
fdf["NAICS_norm"] = fdf["Industry (NAICS) Code"].apply(normalize_naics)

# ── Controls in a row ───────────────────────────────────────────────────
# Industry selection in its own row
available_naics = (
    fdf["NAICS_norm"]
    .unique()
    .tolist()
)
naics_map = wage_df.set_index("NAICS_norm")["NAICS_TITLE"].to_dict()
sel_ind = st.multiselect(
    "Industry(s):",
    options=available_naics,
    format_func=lambda x: naics_map.get(x, x),
    default=available_naics[:3]
)

# Other controls in two columns
c1, c2 = st.columns(2)

with c1:
    wage_metric = st.radio(
        "Wage metric:",
        options=["Annual_Median", "Annual_Pct25–75", "Full (10–90)"],
        index=0
    )

with c2:
    show_counts = st.checkbox("Overlay H‑1B counts", value=False)

st.divider()

# ── Prepare data ────────────────────────────────────────────────────────
# 1) wage subset - only national level
wsub = wage_df[wage_df["NAICS_norm"].isin(sel_ind)]

# 2) H‑1B counts per industry (national level)
h = fdf[fdf["NAICS_norm"].isin(sel_ind)]
counts = (
    h.groupby("NAICS_norm")["Total Applications"]
     .sum()
     .reindex(sel_ind)
     .fillna(0)
     .to_frame("count")
)

# ── Build figure ────────────────────────────────────────────────────────
fig = go.Figure()

# Add a violin (or box) per industry
for naics in sel_ind:
    row = wsub[wsub["NAICS_norm"] == naics]
    if row.empty:
        continue
        
    vals = []
    if wage_metric == "Annual_Median":
        vals = row["Annual_Median"]
    elif wage_metric == "Annual_Pct25–75":
        vals = row[["Annual_Pct25", "Annual_Median", "Annual_Pct75"]].values.flatten()
    else:  # full 10–90
        vals = row[["Annual_Pct10","Annual_Pct25","Annual_Median","Annual_Pct75","Annual_Pct90"]].values.flatten()

    if len(vals) == 0:
        continue

    # Use the NAICS code as fallback if not in mapping
    display_name = naics_map.get(naics, naics)
    
    fig.add_trace(go.Violin(
        x=[display_name] * len(vals),
        y=vals,
        name=display_name,
        box_visible=True,
        meanline_visible=True,
        spanmode="hard",
    ))

# Optionally overlay bar on secondary y‑axis
if show_counts:
    fig.add_trace(go.Bar(
        x=[naics_map.get(n, n) for n in sel_ind],
        y=counts["count"],
        name="H‑1B Apps",
        yaxis="y2",
        opacity=0.3,
    ))
    # add secondary axis
    fig.update_layout(
        yaxis2=dict(
            title="H‑1B Applications",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )

# Final layout tweaks
fig.update_layout(
    title="Salary Distribution by Industry",
    yaxis=dict(title="Annual Wage ($)"),
    xaxis=dict(title="Industry"),
    violingap=0.5,
    height=600,
    showlegend=False  # Remove legend
)

# Apply consistent styling to all figures
fig = apply_consistent_style(fig)

st.subheader("Salary Distribution by Industry")
st.markdown("""
This visualization shows the distribution of annual wages across different industries. Each violin shape represents the salary range for an industry, with wider sections indicating more common salary levels. 
Hover over the violins to see detailed statistics, and use the "Overlay H‑1B counts" option to compare salary distributions with application volumes.
""")

st.plotly_chart(fig, use_container_width=True)

# Create summary table
summary_data = []
for naics in sel_ind:
    row = wsub[wsub["NAICS_norm"] == naics]
    if not row.empty:
        industry_name = naics_map.get(naics, naics)
        h1b_count = counts.loc[naics, "count"] if naics in counts.index else 0
        
        # Convert wage values to numeric before formatting
        median_wage = pd.to_numeric(row['Annual_Median'].iloc[0], errors='coerce')
        pct25_wage = pd.to_numeric(row['Annual_Pct25'].iloc[0], errors='coerce')
        pct75_wage = pd.to_numeric(row['Annual_Pct75'].iloc[0], errors='coerce')
        
        summary_data.append({
            "Industry": industry_name,
            "Median Wage": f"${median_wage:,.0f}" if pd.notnull(median_wage) else "N/A",
            "25th Percentile": f"${pct25_wage:,.0f}" if pd.notnull(pct25_wage) else "N/A",
            "75th Percentile": f"${pct75_wage:,.0f}" if pd.notnull(pct75_wage) else "N/A",
            "H1B Applications": f"{h1b_count:,}"
        })

if summary_data:
    st.markdown("### Detailed Statistics")
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")  # Add divider before the footer

# Add footer at the very end of the app
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    H1B Visa Analysis Dashboard | Created with Streamlit and Plotly
</div>
""", unsafe_allow_html=True)