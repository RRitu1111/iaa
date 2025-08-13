#!/usr/bin/env python3
"""
IAA Feedback System - Cloud Database Manager
Manages cloud database operations, backups, and migrations
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def test_cloud_connection():
    """Test connection to Supabase cloud database"""
    try:
        print("🔗 Testing cloud database connection...")
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"✅ Connected to: {version}")
            
            # Test basic queries
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cur.fetchone()[0]
            print(f"📊 Found {table_count} tables in database")
            
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def backup_database():
    """Create a backup of current database state"""
    try:
        print("💾 Creating database backup...")
        backup_data = {}
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Backup users
                cur.execute("SELECT * FROM users")
                backup_data['users'] = [dict(row) for row in cur.fetchall()]
                
                # Backup departments
                cur.execute("SELECT * FROM departments")
                backup_data['departments'] = [dict(row) for row in cur.fetchall()]
                
                # Backup forms
                cur.execute("SELECT * FROM forms")
                backup_data['forms'] = [dict(row) for row in cur.fetchall()]
                
                # Add metadata
                backup_data['metadata'] = {
                    'backup_date': datetime.utcnow().isoformat(),
                    'database_url': DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'hidden',
                    'total_users': len(backup_data['users']),
                    'total_departments': len(backup_data['departments']),
                    'total_forms': len(backup_data['forms'])
                }
        
        # Save backup to file
        backup_filename = f"database_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        print(f"✅ Backup saved to: {backup_filename}")
        print(f"📊 Backup contains: {backup_data['metadata']['total_users']} users, {backup_data['metadata']['total_departments']} departments, {backup_data['metadata']['total_forms']} forms")
        return backup_filename
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None

def get_database_stats():
    """Get comprehensive database statistics"""
    try:
        print("📊 Gathering database statistics...")
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stats = {}
                
                # Table sizes
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname;
                """)
                stats['table_stats'] = [dict(row) for row in cur.fetchall()]
                
                # Row counts
                tables = ['users', 'departments', 'forms']
                stats['row_counts'] = {}
                for table in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        stats['row_counts'][table] = cur.fetchone()[0]
                    except:
                        stats['row_counts'][table] = 0
                
                # Recent activity
                cur.execute("""
                    SELECT 
                        'users' as table_name,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as last_24h,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as last_7d
                    FROM users
                    UNION ALL
                    SELECT 
                        'forms' as table_name,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as last_24h,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as last_7d
                    FROM forms;
                """)
                stats['activity'] = [dict(row) for row in cur.fetchall()]
        
        print("✅ Database statistics collected")
        return stats
        
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return None

def main():
    """Main function to run database management tasks"""
    print("🌐 IAA Cloud Database Manager")
    print("=" * 50)
    
    # Test connection
    if not test_cloud_connection():
        print("❌ Cannot proceed without database connection")
        return
    
    # Get statistics
    stats = get_database_stats()
    if stats:
        print(f"\n📊 Database Statistics:")
        for table, count in stats['row_counts'].items():
            print(f"   {table}: {count} records")
    
    # Create backup
    backup_file = backup_database()
    if backup_file:
        print(f"\n💾 Backup completed: {backup_file}")
    
    print("\n✅ Cloud database management completed!")

if __name__ == "__main__":
    main()
