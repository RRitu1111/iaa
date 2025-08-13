#!/usr/bin/env python3
"""
Direct Database Cleaner
Directly clears database using the same connection as the backend.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment variables")
    sys.exit(1)

def clear_database_data():
    """Clear all data from database tables while preserving schema"""
    
    print("🗑️  Starting database data cleanup...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        print("✅ Connected to database")
        
        # Get all table names (excluding system tables)
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        if not table_names:
            print("ℹ️  No tables found in the database")
            cursor.close()
            conn.close()
            return
        
        print(f"📋 Found {len(table_names)} tables:")
        for table in table_names:
            print(f"   - {table}")
        
        # Disable foreign key constraints temporarily
        print("\n🔧 Disabling foreign key constraints...")
        cursor.execute("SET session_replication_role = replica;")
        
        # Clear data from all tables
        print("\n🗑️  Clearing data from tables...")
        cleared_count = 0
        
        for table_name in table_names:
            try:
                # Get row count before deletion
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count_before = cursor.fetchone()[0]
                
                if count_before > 0:
                    # Clear the table
                    cursor.execute(f"DELETE FROM {table_name}")
                    
                    # Reset auto-increment sequences if they exist
                    try:
                        cursor.execute(f"ALTER SEQUENCE IF EXISTS {table_name}_id_seq RESTART WITH 1")
                    except:
                        pass  # Sequence might not exist
                    
                    print(f"   ✅ {table_name}: {count_before} rows deleted")
                    cleared_count += 1
                else:
                    print(f"   ⚪ {table_name}: already empty")
                    
            except Exception as e:
                print(f"   ❌ {table_name}: Error - {str(e)}")
        
        # Re-enable foreign key constraints
        print("\n🔧 Re-enabling foreign key constraints...")
        cursor.execute("SET session_replication_role = DEFAULT;")
        
        # Commit all changes
        conn.commit()
        
        # Verify tables are empty
        print("\n🔍 Verifying tables are empty...")
        total_rows = 0
        for table_name in table_names:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            total_rows += count
            if count > 0:
                print(f"   ⚠️  {table_name}: {count} rows remaining")
        
        if total_rows == 0:
            print("   ✅ All tables are now empty")
        else:
            print(f"   ⚠️  {total_rows} total rows remaining across all tables")
        
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Database cleanup completed!")
        print(f"📊 Summary:")
        print(f"   - Tables processed: {len(table_names)}")
        print(f"   - Tables cleared: {cleared_count}")
        print(f"   - Schema preserved: ✅")
        print(f"   - Total remaining rows: {total_rows}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during database cleanup: {str(e)}")
        return False

def confirm_and_clear():
    """Ask for confirmation before clearing database"""
    
    print("⚠️  DATABASE DATA CLEANUP")
    print("=" * 50)
    print("This will DELETE ALL DATA from the database!")
    print("The schema structure will be preserved.")
    print("This action CANNOT be undone.")
    print("=" * 50)
    
    response = input("\nAre you sure you want to proceed? (type 'YES' to confirm): ")
    
    if response.strip().upper() == 'YES':
        return clear_database_data()
    else:
        print("❌ Operation cancelled")
        return False

if __name__ == "__main__":
    success = confirm_and_clear()
    if success:
        print("\n✨ Database cleared! You can now test with fresh data.")
        print("🔄 The backend will reinitialize departments on next startup.")
    else:
        print("\n❌ Cleanup failed.")
