import re
import sqlite3
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/catalogue/page-{}.html"
DB_NAME = "zepto_catalog.db"
GBP_TO_INR_RATE = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

def scrape_books(max_pages=5):
    records = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            title = article.h3.a["title"].strip()
            price_raw = article.find("p", class_="price_color").text.strip()
            rating_classes = article.find("p", class_="star-rating")["class"]
            rating_raw = [c for c in rating_classes if c != "star-rating"][0]
            avail_raw = article.find("p", class_="instock availability").text.strip()
            category = "General"

            records.append({
                "title": title,
                "price_raw": price_raw,
                "rating_raw": rating_raw,
                "availability_raw": avail_raw,
                "category": category
            })

    return pd.DataFrame(records)

def clean_and_transform(df):
    cleaned = df.copy()

    def parse_price(val):
        match = re.search(r"(\d+\.\d+)", str(val))
        return float(match.group(1)) if match else np.nan

    cleaned["price_gbp"] = cleaned["price_raw"].apply(parse_price)
    if cleaned["price_gbp"].isna().any():
        median_val = cleaned["price_gbp"].median()
        cleaned["price_gbp"] = cleaned["price_gbp"].fillna(median_val)

    cleaned["rating"] = cleaned["rating_raw"].map(RATING_MAP).astype(int)
    cleaned["in_stock"] = cleaned["availability_raw"].str.contains("In stock", case=False).astype(int)
    cleaned["price_inr"] = (cleaned["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return cleaned[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]

def setup_database(df, db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute("DROP TABLE IF EXISTS books;")
    cur.execute("DROP TABLE IF EXISTS categories;")

    cur.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    unique_cats = df["category"].unique()
    for cat in unique_cats:
        cur.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    cat_df = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    merged = df.merge(cat_df, left_on="category", right_on="category_name")

    for _, row in merged.iterrows():
        cur.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["title"], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"], row["category_id"]))

    conn.commit()
    return conn

def run_sql_queries(conn):
    queries = {
        "Query 1 (SELECT, WHERE, LIMIT)": """
            SELECT title, price_gbp, rating 
            FROM books 
            WHERE rating >= 4 
            LIMIT 5;
        """,
        "Query 2 (DISTINCT, ORDER BY)": """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating DESC;
        """,
        "Query 3 (BETWEEN, ORDER BY)": """
            SELECT title, price_inr 
            FROM books 
            WHERE price_inr BETWEEN 2000 AND 4000 
            ORDER BY price_inr ASC 
            LIMIT 5;
        """,
        "Query 4 (IN)": """
            SELECT title, rating, in_stock 
            FROM books 
            WHERE rating IN (1, 5) 
            LIMIT 5;
        """,
        "Query 5 (JOIN, ORDER BY, LIMIT)": """
            SELECT b.book_id, b.title, b.price_inr, b.rating, c.category_name
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.in_stock = 1
            ORDER BY b.rating DESC, b.price_inr DESC
            LIMIT 10;
        """
    }

    for name, q in queries.items():
        print(f"\n--- {name} ---")
        print(pd.read_sql(q, conn))

def verify_equivalence(conn):
    join_query = """
        SELECT b.book_id, b.title, b.price_inr, b.rating, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.in_stock = 1
        ORDER BY b.rating DESC, b.price_inr DESC
        LIMIT 10;
    """
    sql_df = pd.read_sql(join_query, conn)

    raw_books = pd.read_sql("SELECT * FROM books", conn)
    raw_categories = pd.read_sql("SELECT * FROM categories", conn)

    merged_df = (
        pd.merge(raw_books, raw_categories, on="category_id")
        .query("in_stock == 1")[["book_id", "title", "price_inr", "rating", "category_name"]]
        .sort_values(by=["rating", "price_inr"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )

    assert sql_df.equals(merged_df), "Equivalence check failed."
    print("\nVerification Passed: SQL JOIN matches pd.merge identically.")

if __name__ == "__main__":
    print("Scraping data...")
    raw_df = scrape_books(max_pages=5)
    print(f"Scraped {len(raw_df)} items.")
    clean_df = clean_and_transform(raw_df)
    db_conn = setup_database(clean_df)
    run_sql_queries(db_conn)
    verify_equivalence(db_conn)
    db_conn.close()
