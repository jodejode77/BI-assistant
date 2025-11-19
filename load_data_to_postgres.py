#!/usr/bin/env python3

import os
import csv
import psycopg2
import sys
from pathlib import Path

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "home_credit")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DATA_DIR = os.getenv("DATA_DIR", "./data")

def get_pg_type(value):
    if value == "" or value is None:
        return "TEXT"
    try:
        int(value)
        return "INTEGER"
    except ValueError:
        pass
    try:
        float(value)
        return "NUMERIC"
    except ValueError:
        pass
    return "TEXT"

def infer_column_types(csv_path, sample_size=1000):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    f = None
    encoding_used = None
    
    for encoding in encodings:
        try:
            test_f = open(csv_path, 'r', encoding=encoding)
            test_f.readline()
            test_f.seek(0)
            test_f.close()
            f = open(csv_path, 'r', encoding=encoding)
            encoding_used = encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if f is None:
        raise ValueError(f"Could not decode {csv_path} with any encoding")
    
    try:
        with f:
            reader = csv.reader(f)
            headers = next(reader)
            
            type_counts = {col: {} for col in headers}
            
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break
                for j, value in enumerate(row):
                    if j < len(headers):
                        col_type = get_pg_type(value)
                        type_counts[headers[j]][col_type] = type_counts[headers[j]].get(col_type, 0) + 1
            
            column_types = {}
            for col in headers:
                if type_counts[col]:
                    most_common = max(type_counts[col], key=type_counts[col].get)
                    column_types[col] = most_common
                else:
                    column_types[col] = "TEXT"
            
            return headers, column_types
    except Exception as e:
        if f:
            f.close()
        raise

def sanitize_name(name):
    return name.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_')

def create_table(conn, table_name, headers, column_types):
    cur = conn.cursor()
    
    columns = []
    for i, col in enumerate(headers):
        if not col or col.strip() == '':
            col_name = f'col_{i}'
        else:
            col_name = sanitize_name(col)
        columns.append(f'"{col_name}" {column_types[col]}')
    
    create_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(columns)})'
    
    cur.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
    cur.execute(create_sql)
    conn.commit()
    cur.close()

def load_csv(conn, csv_path, table_name):
    cur = conn.cursor()
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    f = None
    
    for encoding in encodings:
        try:
            test_f = open(csv_path, 'r', encoding=encoding)
            test_f.readline()
            test_f.seek(0)
            test_f.close()
            f = open(csv_path, 'r', encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if f is None:
        raise ValueError(f"Could not decode {csv_path} with any encoding")
    
    try:
        with f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            sanitized_headers = []
            for i, h in enumerate(headers):
                if not h or h.strip() == '':
                    sanitized_headers.append(f'col_{i}')
                else:
                    sanitized_headers.append(sanitize_name(h))
            placeholders = ', '.join(['%s'] * len(headers))
            columns = ', '.join([f'"{h}"' for h in sanitized_headers])
            
            insert_sql = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
            
            batch = []
            batch_size = 1000
            row_count = 0
            
            for row in reader:
                values = [row[h] if row[h] != '' else None for h in headers]
                batch.append(values)
                
                if len(batch) >= batch_size:
                    cur.executemany(insert_sql, batch)
                    conn.commit()
                    row_count += len(batch)
                    print(f"  Loaded {row_count} rows...", end='\r')
                    batch = []
            
            if batch:
                cur.executemany(insert_sql, batch)
                conn.commit()
                row_count += len(batch)
        
        cur.close()
        return row_count
    except Exception as e:
        if f:
            f.close()
        raise

def main():
    print("Connecting to PostgreSQL...")
    try:
        conn_postgres = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database="postgres",
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn_postgres.autocommit = True
        cur = conn_postgres.cursor()
        
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'")
        if not cur.fetchone():
            print(f"Creating database {DB_NAME}...")
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database {DB_NAME} created!")
        else:
            print(f"Database {DB_NAME} already exists")
        
        cur.close()
        conn_postgres.close()
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected successfully!")
        print("")
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running (docker compose up -d)")
        sys.exit(1)
    
    data_path = Path(DATA_DIR)
    all_csv_files = list(data_path.glob("*.csv"))
    
    ignore_files = ['sample_submission.csv']
    csv_files = [f for f in all_csv_files if f.name not in ignore_files]
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        sys.exit(1)
    
    ignored_count = len(all_csv_files) - len(csv_files)
    if ignored_count > 0:
        print(f"Ignoring {ignored_count} file(s): {', '.join(ignore_files)}")
    
    print(f"Found {len(csv_files)} CSV files to load")
    print("")
    
    for csv_file in csv_files:
        filename = csv_file.stem
        table_name = sanitize_name(filename)
        
        print(f"Processing {filename} -> {table_name}...")
        
        print("  Inferring column types...")
        headers, column_types = infer_column_types(csv_file)
        
        print(f"  Creating table with {len(headers)} columns...")
        create_table(conn, table_name, headers, column_types)
        
        print("  Loading data...")
        row_count = load_csv(conn, csv_file, table_name)
        
        print(f"  ✓ Loaded {row_count} rows into {table_name}")
        print("")
    
    conn.close()
    print("Data loading completed!")

if __name__ == "__main__":
    main()



