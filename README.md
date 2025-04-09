# Road_Accidents_Optimization

This project uses Integer Linear Programming (ILP) to optimally allocate police and awareness resources across districts in Rajasthan to reduce the impact of road accidents.

---

## ⚙️ Requirements

Before running the project, make sure the following Python libraries are installed:

```bash
pip install pandas pulp matplotlib seaborn
```

---

## 📁 Files Needed

Make sure the following files are in the **same folder**:

- `OFS_CODE.py` – Main script  
- `rajasthan_accident_data_2018.csv` – Input dataset  

---

## ▶️ How to Run

Run the Python script using:

```bash
python OFS_CODE.py
```

---

## 📤 Output

- A CSV file `optimized_allocation.csv` will be generated with allocation results.
- The script also prints a summary and displays two plots:
  - Top 10 Districts by Normalized Impact (%)
  - Resource Allocation in Top 10 Districts (stacked bar graph)
