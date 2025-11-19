#!/bin/bash

set -e

export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export DB_NAME="${DB_NAME:-home_credit}"
export DB_USER="${DB_USER:-postgres}"
export DB_PASSWORD="${DB_PASSWORD:-postgres}"
export DATA_DIR="${DATA_DIR:-./data}"

echo "Starting PostgreSQL with Docker..."
docker compose up -d postgres

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install psycopg2-binary
fi

echo "Waiting for PostgreSQL to be ready..."
python3 <<EOF
import psycopg2
import sys
import time
import os

db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', 'postgres')

max_retries = 30
for i in range(max_retries):
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database='postgres'
        )
        conn.close()
        print("PostgreSQL is ready!")
        sys.exit(0)
    except psycopg2.OperationalError:
        if i < max_retries - 1:
            print("PostgreSQL is unavailable - sleeping")
            time.sleep(1)
        else:
            print("PostgreSQL failed to start")
            sys.exit(1)
EOF

echo "Loading data into PostgreSQL using Python script..."
echo ""

python3 load_data_to_postgres.py

echo ""
echo "Data loading completed!"

