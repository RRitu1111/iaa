#!/usr/bin/env python3

"""
Email System Test Script
Tests the supervisor notification email system
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the email function from the backend
try:
    from working_backend import send_supervisor_rating_alert
    print("✅ Successfully imported email function")
except ImportError as e:
    print(f"❌ Failed to import email function: {e}")
    sys.exit(1)

def test_email_system():
    """Test the supervisor notification email system"""
    
    print("🧪 Testing IAA Feedback System Email Notifications")
    print("=" * 60)
    
    # Test data
    trainer_name = "John Doe"
    trainer_email = "john.doe@iaa.edu.in"
    supervisor_email = "supervisor@iaa.edu.in"  # Change this to your test email
    current_rating = 2.5
    form_title = "Flight Training Session - Week 1"
    department_name = "Flight Operations"
    
    print(f"📋 Test Parameters:")
    print(f"   Trainer: {trainer_name} ({trainer_email})")
    print(f"   Supervisor: {supervisor_email}")
    print(f"   Rating: {current_rating}/5.0")
    print(f"   Form: {form_title}")
    print(f"   Department: {department_name}")
    print()
    
    # Check environment variables
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not smtp_username or not smtp_password:
        print("⚠️  Email configuration not found!")
        print("   Please set the following environment variables:")
        print("   - SMTP_USERNAME=your_email@iaa.edu.in")
        print("   - SMTP_PASSWORD=your_app_password")
        print()
        print("📧 Gmail App Password Setup:")
        print("   1. Enable 2-Factor Authentication on Gmail")
        print("   2. Go to Google Account Settings > Security")
        print("   3. Under 2-Step Verification, click 'App passwords'")
        print("   4. Select 'Mail' and 'Other (Custom name)'")
        print("   5. Enter 'IAA Feedback System'")
        print("   6. Copy the 16-character password")
        print()
        return False
    
    print(f"✅ Email configuration found:")
    print(f"   SMTP Username: {smtp_username}")
    print(f"   SMTP Password: {'*' * len(smtp_password)}")
    print()
    
    # Test email sending
    print("📤 Sending test email...")
    try:
        result = send_supervisor_rating_alert(
            trainer_name=trainer_name,
            trainer_email=trainer_email,
            supervisor_email=supervisor_email,
            current_rating=current_rating,
            form_title=form_title,
            department_name=department_name
        )
        
        if result:
            print("✅ Email sent successfully!")
            print(f"📧 Check {supervisor_email} for the notification email")
            print()
            print("📋 Email should contain:")
            print("   - Professional HTML formatting")
            print("   - Trainer performance details")
            print("   - Rating information and threshold")
            print("   - Recommended actions")
            print("   - IAA contact information")
            return True
        else:
            print("❌ Email sending failed!")
            print("   Check your SMTP configuration and credentials")
            return False
            
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

if __name__ == "__main__":
    success = test_email_system()
    
    print()
    print("=" * 60)
    if success:
        print("🎉 Email system test completed successfully!")
        print("🚀 Your IAA Feedback System is ready for production!")
    else:
        print("❌ Email system test failed!")
        print("🔧 Please check your email configuration and try again")
    print("=" * 60)
