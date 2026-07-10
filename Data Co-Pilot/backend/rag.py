"""
rag.py - RAG pipeline using ChromaDB for self-correction.
"""
import chromadb
from chromadb.utils import embedding_functions

DOCS = [
    ("pd_read",     "pd.read_csv(filepath) reads CSV. pd.read_excel() for Excel. pd.read_json() for JSON."),
    ("pd_info",     "df.info() shows dtypes. df.describe() gives stats. df.shape gives (rows, cols). df.head(n) shows first n rows."),
    ("pd_groupby",  "df.groupby('col').agg({'col2':'sum'}) groups data. df.groupby('col')['col2'].mean() for mean."),
    ("pd_filter",   "df[df['col'] > val] filters rows. df[(df['a']>5) & (df['b']=='x')] for multiple conditions."),
    ("pd_sort",     "df.sort_values('col', ascending=False) sorts. df.nlargest(10, 'col') gets top 10."),
    ("pd_null",     "df.isnull().sum() counts nulls. df.dropna() removes nulls. df.fillna(0) fills nulls."),
    ("pd_dtype",    "pd.to_numeric(df['col'], errors='coerce') converts to number. df['col'].astype(str) to string."),
    ("pd_corr",     "df.corr() computes correlation matrix. df[['a','b']].corr() for specific columns."),
    ("pd_value",    "df['col'].value_counts() counts unique values. df['col'].nunique() returns unique count."),
    ("pd_datetime", "pd.to_datetime(df['col']) converts to datetime. df['col'].dt.year/.dt.month/.dt.day for parts."),
    ("plt_bar",     "plt.bar(x, y) creates bar chart. plt.barh for horizontal. Always add title, xlabel, ylabel."),
    ("plt_line",    "plt.plot(x, y, marker='o') creates line chart. Add label= for legend. plt.legend() to show."),
    ("plt_scatter", "plt.scatter(x, y, alpha=0.6) creates scatter. c= for color, s= for size."),
    ("plt_hist",    "plt.hist(df['col'], bins=20) creates histogram. df['col'].hist(bins=20) also works."),
    ("plt_pie",     "plt.pie(vals, labels=labels, autopct='%1.1f%%') creates pie chart."),
    ("sns_heat",    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f') creates heatmap."),
    ("sns_box",     "sns.boxplot(data=df, x='cat', y='num') creates boxplot. sns.violinplot for distribution."),
    ("err_key",     "KeyError: column not found. Use df.columns to check. Column names are case-sensitive."),
    ("err_type",    "TypeError: wrong dtype. Use pd.to_numeric(df['col'], errors='coerce') or astype(float)."),
    ("err_val",     "ValueError in plot: likely non-numeric data. Convert with pd.to_numeric first."),
]

_col = None

def _get_col():
    global _col
    if _col: return _col
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    _col = client.get_or_create_collection("docs", embedding_function=ef)
    if _col.count() == 0:
        _col.add(ids=[d[0] for d in DOCS], documents=[d[1] for d in DOCS])
    return _col

def get_docs(query: str, n: int = 4) -> str:
    try:
        col = _get_col()
        res = col.query(query_texts=[query], n_results=n)
        docs = res["documents"][0] if res["documents"] else []
        return "\n".join(f"- {d}" for d in docs)
    except Exception as e:
        return f"RAG error: {e}"
