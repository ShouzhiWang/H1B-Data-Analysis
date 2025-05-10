
# H1B Visa Analysis Dashboard

An interactive Streamlit application that combines USCIS H‑1B petition data with BLS OEWS wage statistics to explore:

- Geographic patterns (by state & city)  
- Top petitioning industries and employers  
- Application volumes by fiscal year  
- Approval‑rate trends over time, by state or industry  
- Company‑size breakdown of sponsorship  
- Salary distributions across industries, with optional overlay of H‑1B counts  


---

## 🚀 Features

1. **Choropleth Map** of total H‑1B applications by state  
2. **Bar Charts** for the top 10 cities, industries, and companies  
3. **Stacked Bar** of initial vs. continuing filings by year  
4. **Pie Chart** showing company‑size share of petitions  
5. **Line Charts**  
   - Overall approval rate over time  
   - Approval‑rate trends by state or by industry  
6. **Violin Plots** of annual wage distributions by NAICS industry  
   - Toggle between median, IQR (25–75 pct), or full (10–90 pct) range  
   - Optional secondary‐axis overlay of H‑1B petition counts  
7. **Interactive Controls** at the top of each chart (no sidebar), including multi‑select, radio buttons, sliders, and checkboxes  
8. **Auxiliary Text & Expanders** explain how to read each visualization and why it matters  
9. **Consistent Styling** via embedded CSS  
10. **Caching** of data loads for performance  

---

## 📥 Installation

1. Clone this repo
   
2. Create & activate a virtual environment

3. Install dependencies

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Visualizations & Controls

### Filters (Sidebar)

* Fiscal year range slider
* “About H‑1B Application Types” expander
* Data‐source & author info

### Main Panel

1. **Introduction** text under the title
2. **H1B Applications by State**

   * Choropleth map & explanatory caption
3. **Top 10 Cities**

   * Horizontal bar chart & expander with detailed table
4. **Applications by Fiscal Year**

   * Stacked bar of initial vs. continuing cases
5. **Top 10 Industries / Companies**

   * State‐select dropdown & horizontal bars
6. **Company Size Distribution**

   * Pie chart in an expander (with caveat on matching accuracy)
7. **Overall Approval Rate**

   * Line chart of approvals ÷ applications
8. **Approval Rate by State or Industry**

   * Radio button to toggle grouping
   * Multi‑select of top items or “show all” / “show average”
9. **Salary Distribution by Industry**

   * Industry multi‑select, wage‑metric radio, count‑overlay checkbox
   * Violin plot with box & meanline, plus a summary table
   * Expander explaining how to read the violin plot
10. **Footer** with credits & data sources

---

## 🗂️ Data Sources

* **USCIS H‑1B Employer Data Hub**
  [https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub](https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub)
* **BLS OEWS 2023**
  [https://www.bls.gov/oes/](https://www.bls.gov/oes/)
* **Company‑Size Datasets** (e.g. Kaggle)
* **World Cities Geocoder** (SimpleMaps)


---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Created by Shawn Wang*
*— 2025*
