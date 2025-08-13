#!/usr/bin/env python3
"""
Clear Database Data Script
Removes all data from database tables while preserving the schema structure.
"""

import os
import asyncio
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment variables")
    exit(1)

async def clear_database_data():
    """Clear all data from database tables while preserving schema"""
    
    print("🗑️  Starting database data cleanup...")
    print(f"🔗 Connecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'database'}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # Get all table names (excluding system tables)
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        
        tables = await conn.fetch(tables_query)
        table_names = [table['table_name'] for table in tables]
        
        if not table_names:
            print("ℹ️  No tables found in the database")
            await conn.close()
            return
        
        print(f"📋 Found {len(table_names)} tables:")
        for table in table_names:
            print(f"   - {table}")
        
        # Disable foreign key constraints temporarily
        print("\n🔧 Disabling foreign key constraints...")
        await conn.execute("SET session_replication_role = replica;")
        
        # Clear data from all tables
        print("\n🗑️  Clearing data from tables...")
        cleared_count = 0
        
        for table_name in table_names:
            try:
                # Get row count before deletion
                count_before = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                
                if count_before > 0:
                    # Clear the table
                    await conn.execute(f"DELETE FROM {table_name}")
                    
                    # Reset auto-increment sequences if they exist
                    try:
                        await conn.execute(f"ALTER SEQUENCE IF EXISTS {table_name}_id_seq RESTART WITH 1")
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
        await conn.execute("SET session_replication_role = DEFAULT;")
        
        # Verify tables are empty
        print("\n🔍 Verifying tables are empty...")
        total_rows = 0
        for table_name in table_names:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            total_rows += count
            if count > 0:
                print(f"   ⚠️  {table_name}: {count} rows remaining")
        
        if total_rows == 0:
            print("   ✅ All tables are now empty")
        else:
            print(f"   ⚠️  {total_rows} total rows remaining across all tables")
        
        await conn.close()
        
        print(f"\n🎉 Database cleanup completed!")
        print(f"📊 Summary:")
        print(f"   - Tables processed: {len(table_names)}")
        print(f"   - Tables cleared: {cleared_count}")
        print(f"   - Schema preserved: ✅")
        print(f"   - Total remaining rows: {total_rows}")
        
    except Exception as e:
        print(f"❌ Error during database cleanup: {str(e)}")
        raise

async def confirm_and_clear():
    """Ask for confirmation before clearing database"""
    
    print("⚠️  DATABASE DATA CLEANUP")
    print("=" * 50)
    print("This will DELETE ALL DATA from the database!")
    print("The schema structure will be preserved.")
    print("This action CANNOT be undone.")
    print("=" * 50)
    
    response = input("\nAre you sure you want to proceed? (type 'YES' to confirm): ")
    
    if response.strip().upper() == 'YES':
        await clear_database_data()
    else:
        print("❌ Operation cancelled")

if __name__ == "__main__":
    asyncio.run(confirm_and_clear())
