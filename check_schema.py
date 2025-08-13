#!/usr/bin/env python3
"""
Check database schema
"""

import psycopg2

DATABASE_URL = "postgresql://postgres.ltsosvqwyqqzmfnepchd:anshious%232004@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

def check_schema():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    print("Forms table columns:")
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'forms'
        ORDER BY ordinal_position
    """)

    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")
    
    conn.close()

if __name__ == "__main__":
    check_schema()
