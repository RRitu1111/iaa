#!/usr/bin/env python3
"""
IAA Feedback System - Working Production Backend with Cloud Database
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import jwt
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
import time
import asyncio
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import statistics
import re
import random
from collections import Counter
from contextlib import contextmanager
from dotenv import load_dotenv
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Load environment variables
load_dotenv()

# Cloud Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Warning: DATABASE_URL not found in environment variables. Using default configuration.")
    # Construct database URL from individual components for better security
    DB_USER = os.getenv("DB_USER", "postgres.ltsosvqwyqqzmfnepchd")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "anshious#2004")
    DB_HOST = os.getenv("DB_HOST", "aws-0-ap-south-1.pooler.supabase.com")
    DB_PORT = os.getenv("DB_PORT", "6543")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    
    # Construct URL with proper escaping of special characters
    from urllib.parse import quote_plus
    DATABASE_URL = f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
JWT_SECRET = os.getenv("JWT_SECRET", "iaa-production-jwt-secret-2024")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "IAA Feedback System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Email Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@iaa.edu.in")
FROM_NAME = os.getenv("FROM_NAME", "IAA Feedback System")

# Report Configuration
REPORT_THRESHOLD_LOW = float(os.getenv("REPORT_THRESHOLD_LOW", "3.0"))
REPORT_THRESHOLD_MEDIUM = float(os.getenv("REPORT_THRESHOLD_MEDIUM", "3.5"))

# Rating threshold for supervisor notifications
LOW_RATING_THRESHOLD = float(os.getenv("LOW_RATING_THRESHOLD", "3.0"))  # Below 3.0 triggers email
REPORT_AUTO_GENERATE = os.getenv("REPORT_AUTO_GENERATE", "True").lower() == "true"

# FastAPI app
app = FastAPI(
    title=APP_NAME + " API",
    version=APP_VERSION,
    description="Cloud-based feedback system for Indian Aviation Academy",
    debug=DEBUG
)

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode() if body else ""}
    )

# CORS configuration with enhanced error handling and logging
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://iaa-feedback.vercel.app",
    "https://iaa-feedback.vercel.app/",
    # Add any additional domains that need access
]

# Custom CORS middleware with error handling
class CustomCORSMiddleware(CORSMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await super().__call__(scope, receive, send)

        request = Request(scope, receive)
        
        # Log CORS request details
        origin = request.headers.get("origin")
        if origin:
            print(f"🌐 CORS Request from: {origin}")
            print(f"📍 Request path: {request.url.path}")
            print(f"📋 Request method: {scope.get('method', 'UNKNOWN')}")
            
            if origin not in ALLOWED_ORIGINS:
                print(f"⚠️ Warning: Request from non-allowed origin: {origin}")

        return await super().__call__(scope, receive, send)

# Add CORS middleware with enhanced configuration
app.add_middleware(
    CustomCORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"],
    max_age=3600  # Cache preflight requests for 1 hour
)

# Standard CORS middleware is sufficient

# Simple CORS preflight handler
@app.options("/{path:path}")
async def preflight_handler(path: str):
    """Handle CORS preflight requests"""
    return {"message": "OK"}

# Security
security = HTTPBearer()

# CORS preflight handler is defined above

# Cloud Database connection with monitoring
@contextmanager
def get_db():
    """Connect to Supabase PostgreSQL cloud database"""
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            application_name="IAA_Feedback_System"
        )
        yield conn
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database service unavailable: {str(e)}")
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except Exception:
            pass

# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role: str
    department_id: Optional[int] = None
    supervisor_email: Optional[EmailStr] = None

class UserLogin(BaseModel):
    email: Optional[EmailStr] = None  # Optional for backward compatibility
    password: str
    role: Optional[str] = None        # For role-based login
    userId: Optional[str] = None      # For ID-based login

class FormCreate(BaseModel):
    title: str
    description: Optional[str] = None
    form_data: Dict[str, Any]
    department_id: int
    trainer_id: Optional[int] = None
    due_date: Optional[str] = None
    status: Optional[str] = 'draft'  # ✅ Add status field with default 'draft'

class FormDeletionRequest(BaseModel):
    form_id: int
    reason: Optional[str] = None

# Utility functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Analytics and Report Generation Functions
def analyze_sentiment(text: str) -> dict:
    """Simple sentiment analysis based on keywords"""
    if not text or not isinstance(text, str):
        return {"score": 0.0, "sentiment": "neutral"}

    text_lower = text.lower()

    # Positive keywords
    positive_words = [
        'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic', 'outstanding',
        'helpful', 'clear', 'informative', 'engaging', 'useful', 'effective', 'professional',
        'knowledgeable', 'thorough', 'comprehensive', 'well-organized', 'interactive'
    ]

    # Negative keywords
    negative_words = [
        'poor', 'bad', 'terrible', 'awful', 'disappointing', 'confusing', 'unclear',
        'boring', 'useless', 'ineffective', 'unprofessional', 'disorganized', 'rushed',
        'incomplete', 'difficult', 'frustrating', 'waste', 'inadequate'
    ]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    # Calculate sentiment score (-1 to 1)
    total_words = len(text_lower.split())
    if total_words == 0:
        return {"score": 0.0, "sentiment": "neutral"}

    score = (positive_count - negative_count) / max(total_words, 1)

    if score > 0.1:
        sentiment = "positive"
    elif score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"score": score, "sentiment": sentiment}

def analyze_text_themes(texts: list) -> dict:
    """Extract common themes from text responses"""
    if not texts:
        return {"themes": [], "word_frequency": {}}

    # Combine all texts
    combined_text = " ".join([str(text) for text in texts if text]).lower()

    # Remove common stop words and punctuation
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }

    # Extract words (remove punctuation)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    filtered_words = [word for word in words if word not in stop_words]

    # Count word frequency
    word_freq = Counter(filtered_words)

    # Get top themes (most common words)
    top_themes = [word for word, count in word_freq.most_common(10)]

    return {
        "themes": top_themes,
        "word_frequency": dict(word_freq.most_common(20))
    }

def calculate_response_entropy(responses: list) -> float:
    """Calculate entropy to measure response diversity"""
    if not responses:
        return 0.0

    # Count unique responses
    response_counts = Counter([str(r) for r in responses if r is not None])
    total_responses = len(responses)

    if total_responses <= 1:
        return 0.0

    # Calculate entropy
    entropy = 0.0
    for count in response_counts.values():
        probability = count / total_responses
        if probability > 0:
            entropy -= probability * (probability ** 0.5)  # Simplified entropy calculation

    return entropy

def analyze_emoji_sentiment(texts: list) -> dict:
    """Analyze emoji usage in text responses"""
    if not texts:
        return {"emoji_count": 0, "sentiment_emojis": {}}

    # Common emoji patterns
    positive_emojis = ['😊', '😀', '😃', '😄', '😁', '👍', '👌', '💯', '🎉', '✨', '❤️', '💖']
    negative_emojis = ['😞', '😢', '😭', '😠', '😡', '👎', '💔', '😔', '😕', '😟']
    neutral_emojis = ['😐', '😑', '🤔', '😶', '😯']

    combined_text = " ".join([str(text) for text in texts if text])

    emoji_counts = {
        "positive": sum(combined_text.count(emoji) for emoji in positive_emojis),
        "negative": sum(combined_text.count(emoji) for emoji in negative_emojis),
        "neutral": sum(combined_text.count(emoji) for emoji in neutral_emojis)
    }

    total_emojis = sum(emoji_counts.values())

    return {
        "emoji_count": total_emojis,
        "sentiment_emojis": emoji_counts
    }

def generate_pdf_report(report_data: dict) -> bytes:
    """Generate a professional PDF report from report data"""
    try:
        # Create a BytesIO buffer to hold the PDF
        buffer = io.BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        # Get styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2196F3')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1976D2')
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )

        # Story to hold all elements
        story = []

        # Title
        form_title = report_data.get('form', {}).get('title', 'Form Report')
        story.append(Paragraph(f"Comprehensive Report: {form_title}", title_style))
        story.append(Spacer(1, 12))

        # Report metadata
        generated_at = report_data.get('generated_at', '')
        if generated_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%B %d, %Y at %I:%M %p')
                story.append(Paragraph(f"Generated on: {formatted_date}", normal_style))
            except:
                story.append(Paragraph(f"Generated on: {generated_at}", normal_style))

        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))

        stats = report_data.get('statistics', {})
        summary_data = [
            ['Metric', 'Value'],
            ['Total Responses', str(stats.get('total_responses', 0))],
            ['Average Rating', f"{stats.get('average_rating', 0):.2f}/5.0"],
            ['Response Rate', f"{stats.get('response_rate', 0):.1f}%"],
            ['Overall Sentiment', report_data.get('sentiment_analysis', {}).get('sentiment', 'N/A').title()]
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Rating Distribution
        if stats.get('rating_distribution'):
            story.append(Paragraph("Rating Distribution", heading_style))

            rating_data = [['Rating', 'Count', 'Percentage']]
            total_responses = stats.get('total_responses', 1)

            for rating, count in sorted(stats['rating_distribution'].items()):
                percentage = (count / total_responses) * 100
                rating_data.append([f"{rating} stars", str(count), f"{percentage:.1f}%"])

            rating_table = Table(rating_data, colWidths=[1.5*inch, 1*inch, 1.5*inch])
            rating_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(rating_table)
            story.append(Spacer(1, 20))

        # Sentiment Analysis
        sentiment = report_data.get('sentiment_analysis', {})
        if sentiment:
            story.append(Paragraph("Sentiment Analysis", heading_style))

            sentiment_text = f"""
            <b>Overall Sentiment:</b> {sentiment.get('sentiment', 'N/A').title()}<br/>
            <b>Sentiment Score:</b> {sentiment.get('score', 0):.3f}<br/>
            <br/>
            The sentiment analysis reveals the overall emotional tone of the feedback responses.
            A positive sentiment indicates satisfaction, while negative sentiment suggests areas for improvement.
            """

            story.append(Paragraph(sentiment_text, normal_style))
            story.append(Spacer(1, 15))

        # Text Themes
        themes = report_data.get('text_themes', {})
        if themes.get('themes'):
            story.append(Paragraph("Common Themes", heading_style))

            themes_text = "The following themes were identified from text responses:<br/><br/>"
            for i, theme in enumerate(themes['themes'][:10], 1):
                themes_text += f"{i}. {theme.title()}<br/>"

            story.append(Paragraph(themes_text, normal_style))
            story.append(Spacer(1, 15))

        # Insights and Recommendations
        insights = report_data.get('insights', [])
        recommendations = report_data.get('recommendations', [])

        if insights:
            story.append(Paragraph("Key Insights", heading_style))
            insights_text = ""
            for insight in insights:
                insights_text += f"• {insight}<br/>"
            story.append(Paragraph(insights_text, normal_style))
            story.append(Spacer(1, 15))

        if recommendations:
            story.append(Paragraph("Recommendations", heading_style))
            rec_text = ""
            for rec in recommendations:
                rec_text += f"• {rec}<br/>"
            story.append(Paragraph(rec_text, normal_style))
            story.append(Spacer(1, 15))

        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        story.append(Paragraph("Generated by IAA Feedback System", footer_style))

        # Build PDF
        doc.build(story)

        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    except Exception as e:

        raise Exception(f"Failed to generate PDF: {str(e)}")

def send_email_alert(to_email: str, subject: str, body: str, attachment_path: str = None) -> bool:
    """Send email alert to administrators"""
    if not SMTP_USERNAME or not SMTP_PASSWORD:

        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)

        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(FROM_EMAIL, to_email, text)
        server.quit()

        return True
    except Exception as e:
        return False

def send_supervisor_rating_alert(trainer_name: str, trainer_email: str, supervisor_email: str,
                               current_rating: float, form_title: str, department_name: str = "") -> bool:
    """Send email alert to supervisor when trainer rating falls below threshold"""
    if not supervisor_email or not SMTP_USERNAME or not SMTP_PASSWORD:
        return False

    subject = f"⚠️ Low Performance Alert - {trainer_name}"

    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .details {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d; }}
            .rating {{ font-size: 18px; font-weight: bold; color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🛩️ IAA Feedback System - Performance Alert</h2>
            <p>This is an automated notification regarding trainer performance.</p>
        </div>

        <div class="alert">
            <h3>⚠️ Low Rating Alert</h3>
            <p>A trainer under your supervision has received a rating below the acceptable threshold.</p>
        </div>

        <div class="details">
            <h4>📋 Details:</h4>
            <ul>
                <li><strong>Trainer Name:</strong> {trainer_name}</li>
                <li><strong>Trainer Email:</strong> {trainer_email}</li>
                <li><strong>Department:</strong> {department_name}</li>
                <li><strong>Form/Session:</strong> {form_title}</li>
                <li><strong>Current Rating:</strong> <span class="rating">{current_rating:.1f}/5.0</span></li>
                <li><strong>Threshold:</strong> {LOW_RATING_THRESHOLD}/5.0</li>
                <li><strong>Date:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</li>
            </ul>
        </div>

        <div class="details">
            <h4>📝 Recommended Actions:</h4>
            <ul>
                <li>Schedule a one-on-one meeting with the trainer</li>
                <li>Review the specific feedback and areas for improvement</li>
                <li>Provide additional training or support if needed</li>
                <li>Monitor future performance closely</li>
            </ul>
        </div>

        <div class="footer">
            <p>This email was sent automatically by the IAA Feedback System.</p>
            <p>If you have any questions, please contact the system administrator.</p>
            <p><strong>Indian Aviation Academy</strong><br>
            Email: admin@iaa.edu.in | Phone: +91-80-2522-3001</p>
        </div>
    </body>
    </html>
    """

    return send_email_alert(supervisor_email, subject, body)

def generate_comprehensive_report(form_id: int) -> dict:
    """Generate comprehensive analytics report for a form"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get form details
                cur.execute("""
                    SELECT f.*, u.first_name || ' ' || u.last_name as creator_name,
                           d.name as department_name
                    FROM forms f
                    JOIN users u ON COALESCE(f.creator_id, f.trainer_id) = u.id
                    JOIN departments d ON f.department_id = d.id
                    WHERE f.id = %s
                """, (form_id,))

                form = cur.fetchone()
                if not form:
                    return {"error": "Form not found"}

                form_dict = dict(form)

                # Get all responses
                cur.execute("""
                    SELECT fr.*, u.first_name || ' ' || u.last_name as user_name,
                           u.email as user_email, d.name as department_name
                    FROM form_responses fr
                    JOIN users u ON fr.user_id = u.id
                    JOIN departments d ON u.department_id = d.id
                    WHERE fr.form_id = %s
                """, (form_id,))

                responses = cur.fetchall()

                if not responses:
                    return {
                        "form": form_dict,
                        "total_responses": 0,
                        "message": "No responses to analyze"
                    }

                # Process responses for analysis
                all_ratings = []
                all_text_responses = []
                response_data = []

                for response in responses:
                    response_dict = dict(response)
                    if isinstance(response_dict["response_data"], str):
                        try:
                            response_dict["response_data"] = json.loads(response_dict["response_data"])
                        except:
                            response_dict["response_data"] = {}

                    response_data.append(response_dict)

                    # Extract ratings and text
                    for key, value in response_dict["response_data"].items():
                        if isinstance(value, (int, float)) and 1 <= value <= 5:
                            all_ratings.append(value)
                        elif isinstance(value, str) and len(value.strip()) > 0:
                            all_text_responses.append(value.strip())

                # Calculate statistics
                stats = {
                    "total_responses": len(responses),
                    "response_rate": min(100, (len(responses) / 50) * 100),  # Assuming target of 50
                    "average_rating": statistics.mean(all_ratings) if all_ratings else 0,
                    "median_rating": statistics.median(all_ratings) if all_ratings else 0,
                    "rating_distribution": dict(Counter(all_ratings)) if all_ratings else {},
                    "completion_time": "N/A"  # Could be calculated if we track start/end times
                }

                # Perform advanced analytics
                sentiment_analysis = analyze_sentiment(" ".join(all_text_responses))
                text_themes = analyze_text_themes(all_text_responses)
                emoji_analysis = analyze_emoji_sentiment(all_text_responses)
                response_entropy = calculate_response_entropy(all_ratings)

                # Generate insights and recommendations
                insights = []
                recommendations = []

                if stats["average_rating"] < REPORT_THRESHOLD_LOW:
                    insights.append("⚠️ Low average rating indicates significant issues")
                    recommendations.append("Immediate review and improvement needed")
                elif stats["average_rating"] < REPORT_THRESHOLD_MEDIUM:
                    insights.append("⚡ Average rating suggests room for improvement")
                    recommendations.append("Consider targeted improvements")
                else:
                    insights.append("✅ Good average rating indicates positive reception")
                    recommendations.append("Maintain current quality standards")

                if sentiment_analysis["sentiment"] == "negative":
                    insights.append("📉 Negative sentiment detected in text responses")
                    recommendations.append("Address specific concerns mentioned in feedback")
                elif sentiment_analysis["sentiment"] == "positive":
                    insights.append("📈 Positive sentiment in text responses")
                    recommendations.append("Identify and replicate successful elements")

                if response_entropy < 0.3:
                    insights.append("🔄 Low response diversity - responses are very similar")
                    recommendations.append("Consider more varied question types")
                elif response_entropy > 0.7:
                    insights.append("🌟 High response diversity - varied feedback received")
                    recommendations.append("Analyze different response patterns")

                # Compile final report
                report = {
                    "form": form_dict,
                    "statistics": stats,
                    "sentiment_analysis": sentiment_analysis,
                    "text_themes": text_themes,
                    "emoji_analysis": emoji_analysis,
                    "response_entropy": response_entropy,
                    "insights": insights,
                    "recommendations": recommendations,
                    "responses": response_data,
                    "generated_at": datetime.utcnow().isoformat(),
                    "report_version": "1.0"
                }

                return report

    except Exception as e:

        return {"error": str(e)}

async def check_and_generate_reports():
    """Background task to check for forms that need reports and generate them"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Find forms that are published and have responses but no recent reports
                cur.execute("""
                    SELECT f.id, f.title, f.due_date, COUNT(fr.id) as response_count
                    FROM forms f
                    LEFT JOIN form_responses fr ON f.id = fr.form_id
                    WHERE (f.status = 'published' OR f.is_published = true)
                    AND (f.due_date IS NULL OR f.due_date < CURRENT_TIMESTAMP)
                    GROUP BY f.id, f.title, f.due_date
                    HAVING COUNT(fr.id) > 0
                """)

                forms_needing_reports = cur.fetchall()

                for form in forms_needing_reports:
                    form_dict = dict(form)

                    # Generate report
                    report = generate_comprehensive_report(form_dict['id'])

                    if "error" not in report:
                        # Check if average rating is below threshold
                        avg_rating = report.get("statistics", {}).get("average_rating", 5.0)

                        if avg_rating < REPORT_THRESHOLD_LOW:
                            # Send alert email to administrators
                            await send_low_score_alert(form_dict, report)

                        # Store report in database (you could add a reports table)
                        pass
                    else:
                        pass

    except Exception as e:
        pass

async def send_low_score_alert(form_data: dict, report: dict):
    """Send email alert for low scoring forms"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get admin emails
                cur.execute("SELECT email FROM users WHERE role = 'admin'")
                admin_emails = [row['email'] for row in cur.fetchall()]

                if not admin_emails:

                    return

                avg_rating = report.get("statistics", {}).get("average_rating", 0)
                total_responses = report.get("statistics", {}).get("total_responses", 0)

                subject = f"🚨 Low Score Alert: {form_data['title']}"

                body = f"""
                <html>
                <body>
                    <h2>Low Score Alert - IAA Feedback System</h2>
                    <p>A form has received scores below the configured threshold and requires attention.</p>

                    <h3>Form Details:</h3>
                    <ul>
                        <li><strong>Form:</strong> {form_data['title']}</li>
                        <li><strong>Average Rating:</strong> {avg_rating:.2f}/5.0</li>
                        <li><strong>Total Responses:</strong> {total_responses}</li>
                        <li><strong>Threshold:</strong> {REPORT_THRESHOLD_LOW}</li>
                    </ul>

                    <h3>Key Insights:</h3>
                    <ul>
                        {''.join([f'<li>{insight}</li>' for insight in report.get('insights', [])])}
                    </ul>

                    <h3>Recommendations:</h3>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in report.get('recommendations', [])])}
                    </ul>

                    <p>Please review the detailed feedback and take appropriate action.</p>

                    <p><em>This is an automated alert from the IAA Feedback System.</em></p>
                </body>
                </html>
                """

                # Send to all admins
                for email in admin_emails:
                    send_email_alert(email, subject, body)

    except Exception as e:
        pass

# Background task scheduler
def start_background_scheduler():
    """Start background task scheduler"""
    def run_scheduler():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def scheduler():
            while True:
                try:
                    if REPORT_AUTO_GENERATE:
                        await check_and_generate_reports()
                    await asyncio.sleep(3600)  # Check every hour
                except Exception as e:

                    await asyncio.sleep(300)  # Wait 5 minutes on error

        loop.run_until_complete(scheduler())

    if REPORT_AUTO_GENERATE:
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data["id"],
        "email": user_data["email"],
        "role": user_data["role"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (payload["user_id"],))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return dict(user)

# Initialize database
def init_database():
    """Initialize the database schema and create default records"""
    print("Initializing database...")
    try:
        with get_db() as conn:
            conn.autocommit = False  # Start transaction mode
            with conn.cursor() as cur:
            # Create tables

            # Departments table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS departments (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    code VARCHAR(20) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    department_id INTEGER REFERENCES departments(id),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)

            # Remove foreign key constraint from department_id to allow any numeric value
            try:
                cur.execute("""
                    ALTER TABLE users DROP CONSTRAINT IF EXISTS users_department_id_fkey
                """)
                cur.execute("""
                    ALTER TABLE users DROP CONSTRAINT IF EXISTS fk_users_department
                """)

            except Exception as e:
                pass

            # Check if users table has required columns, if not add them
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'hashed_password'
            """)
            if not cur.fetchone():

                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255)")

            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'role'
            """)
            if not cur.fetchone():

                cur.execute("ALTER TABLE users ADD COLUMN role VARCHAR(20)")

            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'is_active'
            """)
            if not cur.fetchone():

                cur.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE")

            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'last_login'
            """)
            if not cur.fetchone():

                cur.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP")

            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'has_selected_departments'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE users ADD COLUMN has_selected_departments BOOLEAN DEFAULT FALSE")

            # Check if users table has supervisor_email column, if not add it
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'supervisor_email'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE users ADD COLUMN supervisor_email VARCHAR(255)")

            # Forms table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forms (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    creator_id INTEGER REFERENCES users(id) NOT NULL,
                    department_id INTEGER REFERENCES departments(id) NOT NULL,
                    trainer_id INTEGER REFERENCES users(id),
                    form_data JSONB NOT NULL,
                    status VARCHAR(20) DEFAULT 'draft',
                    type VARCHAR(20) DEFAULT 'single-use',
                    due_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    published_at TIMESTAMP
                )
            """)

            # Add due_date column if it doesn't exist (for existing databases)
            cur.execute("""
                ALTER TABLE forms
                ADD COLUMN IF NOT EXISTS due_date TIMESTAMP
            """)

            # Add status column if it doesn't exist and migrate from is_published
            cur.execute("""
                ALTER TABLE forms
                ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'draft'
            """)

            # Add creator_id column if it doesn't exist and migrate from trainer_id
            cur.execute("""
                ALTER TABLE forms
                ADD COLUMN IF NOT EXISTS creator_id INTEGER REFERENCES users(id)
            """)

            # Migrate existing data from trainer_id to creator_id if creator_id is null
            cur.execute("""
                UPDATE forms
                SET creator_id = trainer_id
                WHERE creator_id IS NULL AND trainer_id IS NOT NULL
            """)

            # Migrate existing data from is_published to status
            cur.execute("""
                UPDATE forms
                SET status = CASE
                    WHEN is_published = true THEN 'published'
                    ELSE 'draft'
                END
                WHERE status IS NULL OR status = 'draft'
            """)

            # Add form_request_id column to forms table if it doesn't exist
            cur.execute("""
                ALTER TABLE forms
                ADD COLUMN IF NOT EXISTS form_request_id INTEGER REFERENCES form_requests(id)
            """)

            # Add form_created columns to form_requests table if they don't exist
            cur.execute("""
                ALTER TABLE form_requests
                ADD COLUMN IF NOT EXISTS form_created BOOLEAN DEFAULT FALSE
            """)

            cur.execute("""
                ALTER TABLE form_requests
                ADD COLUMN IF NOT EXISTS form_created_at TIMESTAMP
            """)

            # Update form_responses table to allow nullable user_id and add submitted_at
            try:
                cur.execute("""
                    ALTER TABLE form_responses
                    ALTER COLUMN user_id DROP NOT NULL
                """)
                cur.execute("""
                    ALTER TABLE form_responses
                    ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
                cur.execute("""
                    UPDATE form_responses
                    SET submitted_at = created_at
                    WHERE submitted_at IS NULL
                """)
            except Exception as e:
                pass

            # Add missing columns to forms table if they don't exist
            try:
                cur.execute("ALTER TABLE forms ADD COLUMN IF NOT EXISTS type VARCHAR(20) DEFAULT 'single-use'")
                cur.execute("ALTER TABLE forms ADD COLUMN IF NOT EXISTS is_published BOOLEAN DEFAULT FALSE")
            except Exception as e:
                pass

            # Form responses table - preserve existing data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS form_responses (
                    id SERIAL PRIMARY KEY,
                    form_id INTEGER REFERENCES forms(id) ON DELETE CASCADE NOT NULL,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    response_data JSONB NOT NULL,
                    is_anonymous BOOLEAN DEFAULT FALSE,
                    is_complete BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Form requests table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS form_requests (
                    id SERIAL PRIMARY KEY,
                    trainer_id INTEGER REFERENCES users(id) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    department_id INTEGER REFERENCES departments(id),
                    session_name VARCHAR(200),
                    session_date DATE,
                    session_duration INTEGER DEFAULT 60,
                    form_validity_duration INTEGER DEFAULT 7,
                    form_type VARCHAR(50) DEFAULT 'feedback',
                    priority VARCHAR(20) DEFAULT 'normal',
                    additional_notes TEXT,
                    status VARCHAR(20) DEFAULT 'pending',
                    admin_response TEXT,
                    reviewed_by INTEGER REFERENCES users(id),
                    reviewed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Form deletion requests table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS form_deletion_requests (
                    id SERIAL PRIMARY KEY,
                    form_id INTEGER REFERENCES forms(id) NOT NULL,
                    trainer_id INTEGER REFERENCES users(id) NOT NULL,
                    reason TEXT,
                    status VARCHAR(20) DEFAULT 'pending',
                    admin_response TEXT,
                    reviewed_by INTEGER REFERENCES users(id),
                    reviewed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add form_data and due_date columns to form_requests if they don't exist
            try:
                cur.execute("""
                    ALTER TABLE form_requests
                    ADD COLUMN IF NOT EXISTS form_data JSONB,
                    ADD COLUMN IF NOT EXISTS due_date TIMESTAMP
                """)

            except Exception as e:
                pass

            # Reports settings table for configuration
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reports_settings (
                    id SERIAL PRIMARY KEY,
                    setting_name VARCHAR(100) UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert default report generation delay setting
            cur.execute("""
                INSERT INTO reports_settings (setting_name, setting_value, description)
                VALUES ('auto_generation_delay_days', '1', 'Number of days after due date to automatically generate reports')
                ON CONFLICT (setting_name) DO NOTHING
            """)

            # Add new columns to existing form_requests table if they don't exist
            try:
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS department_id INTEGER REFERENCES departments(id)")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS session_name VARCHAR(200)")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS session_date DATE")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS session_duration INTEGER DEFAULT 60")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS form_validity_duration INTEGER DEFAULT 7")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS form_type VARCHAR(50) DEFAULT 'feedback'")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal'")
                cur.execute("ALTER TABLE form_requests ADD COLUMN IF NOT EXISTS additional_notes TEXT")

            except Exception as e:

                pass

            # Remove ALL problematic columns that cause NOT NULL constraint errors
            try:
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS purpose")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS target_department")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS question_types")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS duration")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS submission_limit")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS response_limit")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS auto_close")
                cur.execute("ALTER TABLE form_requests DROP COLUMN IF EXISTS notification_settings")

            except Exception as e:

                pass

            # Check if departments table has code column, if not add it
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'departments' AND column_name = 'code'
            """)
            if not cur.fetchone():

                cur.execute("ALTER TABLE departments ADD COLUMN code VARCHAR(20)")
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS departments_code_unique ON departments(code)")

            # Insert default departments
            cur.execute("""
                INSERT INTO departments (name, code, description) VALUES
                ('Flight Operations', 'DEPT001', 'Flight training and operations'),
                ('Aircraft Maintenance', 'DEPT002', 'Aircraft maintenance and engineering'),
                ('Air Traffic Control', 'DEPT003', 'Air traffic control operations'),
                ('Ground Operations', 'DEPT004', 'Ground handling and operations'),
                ('Aviation Safety', 'DEPT005', 'Safety and security operations'),
                ('Cabin Crew', 'DEPT006', 'Cabin crew training and operations')
                ON CONFLICT (code) DO NOTHING
            """)

            # Insert default admin user

            admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")  # CHANGE IN PRODUCTION
            hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            cur.execute("""
                INSERT INTO users (email, first_name, last_name, hashed_password, role, department_id, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    hashed_password = EXCLUDED.hashed_password,
                    role = EXCLUDED.role,
                    department_id = EXCLUDED.department_id,
                    is_active = EXCLUDED.is_active
            """, (
                'admin@iaa.edu.in',
                'System',
                'Administrator',
                hashed_password,
                'admin',
                1,  # Default to first department
                True
            ))

            try:
                conn.commit()
                print("Database changes committed successfully")
            except Exception as commit_error:
                conn.rollback()
                print(f"Error committing changes: {str(commit_error)}")
                raise

# Initialize database on startup
def startup_handler():
    """Handle database initialization during application startup"""
    try:
        print("Starting database initialization...")
        init_database()
        print("Database initialization completed successfully")
    except HTTPException as he:
        print(f"HTTP error during startup: {str(he)}")
        raise he
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # Log the full error details
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize database: {str(e)}"
        )

# Call startup handler when module is imported
startup_handler()

# Routes
@app.get("/")
async def root():
    return {
        "message": "IAA Feedback System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Enhanced health check endpoint that verifies core system components"""
    health_status = {
        "status": "healthy",
        "components": {
            "database": {"status": "unhealthy", "message": "Not checked"},
            "api": {"status": "healthy", "message": "API is responding"},
            "cors": {"status": "healthy", "message": "CORS is configured"}
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # Test database connection with timeout
        start_time = time.time()
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_response_time = time.time() - start_time
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "message": f"Connected and responding ({db_response_time:.2f}s)",
                    "responseTime": db_response_time
                }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # Add version information
    health_status["version"] = APP_VERSION
    health_status["environment"] = ENVIRONMENT

    # Set appropriate status code
    if health_status["status"] == "unhealthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )

    return health_status

@app.get("/db-info")
async def db_info():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users")
                user_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM forms")
                form_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM departments")
                dept_count = cur.fetchone()[0]

        return {
            "success": True,
            "database": {
                "connected": True,
                "users": user_count,
                "forms": form_count,
                "departments": dept_count,
                "status": "connected"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Database connection failed",
            "error": str(e)
        }

@app.get("/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get real-time dashboard statistics for admin and trainer dashboards"""

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if current_user["role"] == "admin":
                    # Admin dashboard stats - optimized single query
                    cur.execute("""
                        SELECT
                            (SELECT COUNT(*) FROM forms) as total_forms,
                            (SELECT COUNT(*) FROM forms WHERE status = 'published' OR is_published = true) as active_forms,
                            (SELECT COUNT(*) FROM forms WHERE status = 'draft' OR is_published = false) as draft_forms,
                            (SELECT COUNT(*) FROM form_responses) as total_responses,
                            (SELECT COUNT(*) FROM form_requests WHERE status = 'pending') as pending_requests,
                            (SELECT COUNT(*) FROM departments) as total_departments,
                            (SELECT COUNT(*) FROM users WHERE role = 'trainer') as total_trainers
                    """)

                    stats = cur.fetchone()

                    return {
                        "success": True,
                        "stats": {
                            "totalForms": stats['total_forms'],
                            "activeForms": stats['active_forms'],
                            "draftForms": stats['draft_forms'],
                            "totalResponses": stats['total_responses'],
                            "pendingRequests": stats['pending_requests'],
                            "totalDepartments": stats['total_departments'],
                            "totalTrainers": stats['total_trainers']
                        }
                    }

                elif current_user["role"] == "trainer":
                    # Trainer dashboard stats - optimized single query
                    cur.execute("""
                        SELECT
                            (SELECT COUNT(*) FROM forms WHERE trainer_id = %s OR creator_id = %s) as my_forms,
                            (SELECT COUNT(*) FROM forms WHERE (trainer_id = %s OR creator_id = %s) AND (status = 'published' OR is_published = true)) as active_forms,
                            (SELECT COUNT(*) FROM forms WHERE (trainer_id = %s OR creator_id = %s) AND (status = 'draft' OR is_published = false)) as draft_forms,
                            (SELECT COUNT(*) FROM form_responses fr JOIN forms f ON fr.form_id = f.id WHERE f.trainer_id = %s OR f.creator_id = %s) as total_responses,
                            (SELECT COUNT(*) FROM form_requests WHERE trainer_id = %s AND status = 'pending') as pending_requests
                    """, (current_user["id"], current_user["id"], current_user["id"], current_user["id"],
                          current_user["id"], current_user["id"], current_user["id"], current_user["id"], current_user["id"]))

                    stats = cur.fetchone()

                    # Calculate average rating for trainer's forms (simplified for performance)
                    cur.execute(r"""
                        SELECT AVG(CAST(fr.response_data->>'rating' AS FLOAT)) as avg_rating
                        FROM form_responses fr
                        JOIN forms f ON fr.form_id = f.id
                        WHERE (f.trainer_id = %s OR f.creator_id = %s)
                        AND fr.response_data->>'rating' IS NOT NULL
                        AND fr.response_data->>'rating' ~ '^[0-9]+\.?[0-9]*$'
                    """, (current_user["id"], current_user["id"]))

                    avg_rating_result = cur.fetchone()
                    avg_rating = avg_rating_result['avg_rating'] if avg_rating_result and avg_rating_result['avg_rating'] else 0.0

                    return {
                        "success": True,
                        "stats": {
                            "myForms": stats['my_forms'],
                            "activeForms": stats['active_forms'],
                            "draftForms": stats['draft_forms'],
                            "totalResponses": stats['total_responses'],
                            "pendingRequests": stats['pending_requests'],
                            "averageRating": round(float(avg_rating), 1) if avg_rating else 0.0
                        }
                    }

                else:
                    # Trainee stats

                    if not current_user.get("department_id"):

                        return {
                            "success": True,
                            "stats": {
                                "availableForms": 0,
                                "completedForms": 0,
                                "pendingForms": 0
                            }
                        }

                    cur.execute("""
                        SELECT COUNT(*) as count FROM forms f
                        WHERE f.department_id = %s AND f.is_published = true
                    """, (current_user["department_id"],))
                    available_forms = cur.fetchone()['count']

                    cur.execute("""
                        SELECT COUNT(*) as count FROM form_responses fr
                        JOIN forms f ON fr.form_id = f.id
                        WHERE fr.user_id = %s
                    """, (current_user["id"],))
                    completed_forms = cur.fetchone()['count']

                    return {
                        "success": True,
                        "stats": {
                            "availableForms": available_forms,
                            "completedForms": completed_forms,
                            "pendingForms": available_forms - completed_forms
                        }
                    }

    except Exception as e:

        import traceback

        raise HTTPException(status_code=500, detail=f"Failed to get dashboard stats: {str(e)}")

@app.post("/setup")
async def setup():
    try:
        init_database()
        return {
            "success": True,
            "message": "Database setup completed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Database setup failed",
            "error": str(e)
        }

# Authentication endpoints
@app.post("/auth/register")
async def register(user_data: UserRegister):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if user exists
                cur.execute("SELECT id FROM users WHERE email = %s", (user_data.email,))
                if cur.fetchone():

                    raise HTTPException(status_code=400, detail="Email already registered")

                # Hash password
                hashed_password = hash_password(user_data.password)

                # Insert user
                cur.execute("""
                    INSERT INTO users (email, first_name, last_name, hashed_password, role, department_id, supervisor_email, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id, email, first_name, last_name, role, department_id, supervisor_email, is_active, created_at
                """, (user_data.email, user_data.first_name, user_data.last_name,
                     hashed_password, user_data.role, user_data.department_id, user_data.supervisor_email, True))

                user = dict(cur.fetchone())
                conn.commit()

                # Create token
                token = create_token(user)

                return {
                    "success": True,
                    "message": "User registered successfully",
                    "data": {
                        "user": {
                            "id": user["id"],
                            "email": user["email"],
                            "first_name": user["first_name"],
                            "last_name": user["last_name"],
                            "role": user["role"],
                            "department_id": user["department_id"],
                            "is_active": user["is_active"],
                            "created_at": user["created_at"].isoformat() if user["created_at"] else None
                        },
                        "access_token": token,
                        "refresh_token": token,  # Same for simplicity
                        "token_type": "bearer"
                    }
                }
    except HTTPException as he:

        raise he
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/auth/login")
async def login(login_data: UserLogin):
    # Clean logging - only show what's actually being used
    if login_data.role and login_data.userId:
        pass  # ID-based login
    elif login_data.email:
        pass  # Email-based login
    else:
        pass  # Invalid format

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                user = None

                # Handle ID-based login (your preferred method)
                if login_data.role and login_data.userId:

                    # Convert userId to email format for lookup (userId@iaa.edu.in)
                    expected_email = f"{login_data.userId}@iaa.edu.in"

                    # Look for user by exact email and role match
                    cur.execute("SELECT * FROM users WHERE email = %s AND role = %s", (expected_email, login_data.role))
                    user = cur.fetchone()

                    if not user:

                        # Check if user exists with different role
                        cur.execute("SELECT role FROM users WHERE email = %s", (expected_email,))
                        existing_user = cur.fetchone()
                        if existing_user:
                            raise HTTPException(status_code=401, detail=f"Wrong role selected. This ID is registered as {existing_user['role']}, not {login_data.role}.")
                        else:
                            raise HTTPException(status_code=401, detail=f"User ID '{login_data.userId}' is not registered. Please register first.")

                    # Verify password against stored hash
                    if not verify_password(login_data.password, user["hashed_password"]):

                        raise HTTPException(status_code=401, detail="Wrong password. Please check your password and try again.")

                # Handle email-based login (backward compatibility)
                elif login_data.email:

                    cur.execute("SELECT * FROM users WHERE email = %s", (login_data.email,))
                    user = cur.fetchone()

                    if not user:

                        raise HTTPException(status_code=401, detail="Email address not found. Please check your email or register first.")

                    if not verify_password(login_data.password, user["password_hash"]):

                        raise HTTPException(status_code=401, detail="Incorrect password. Please check your password and try again.")

                if not user:

                    raise HTTPException(status_code=401, detail="Login failed. Please check your credentials and try again.")

                if not user["is_active"]:

                    raise HTTPException(status_code=403, detail="Your account has been deactivated. Please contact the administrator for assistance.")

                # Update last login
                cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", (user["id"],))
                conn.commit()

                user_dict = dict(user)
                token = create_token(user_dict)

                return {
                    "success": True,
                    "message": "Login successful",
                    "data": {
                        "user": {
                            "id": user_dict["id"],
                            "email": user_dict["email"],
                            "first_name": user_dict["first_name"],
                            "last_name": user_dict["last_name"],
                            "role": user_dict["role"],
                            "department_id": user_dict["department_id"],
                            "is_active": user_dict["is_active"],
                            "has_selected_departments": user_dict.get("has_selected_departments", False),
                            "created_at": user_dict["created_at"].isoformat() if user_dict["created_at"] else None,
                            "last_login": datetime.utcnow().isoformat()
                        },
                        "access_token": token,
                        "refresh_token": token,
                        "token_type": "bearer"
                    }
                }
    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    try:
        payload = verify_token(refresh_token)

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (payload["user_id"],))
                user = cur.fetchone()

                if not user or not user["is_active"]:
                    raise HTTPException(status_code=401, detail="User not found or inactive")

                user_dict = dict(user)
                new_token = create_token(user_dict)

                return {
                    "success": True,
                    "message": "Token refreshed",
                    "data": {
                        "access_token": new_token,
                        "refresh_token": new_token,
                        "token_type": "bearer"
                    }
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {str(e)}")

@app.get("/users/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "first_name": current_user["first_name"],
        "last_name": current_user["last_name"],
        "role": current_user["role"],
        "department_id": current_user["department_id"],
        "is_active": current_user["is_active"],
        "created_at": current_user["created_at"].isoformat() if current_user["created_at"] else None,
        "last_login": current_user["last_login"].isoformat() if current_user["last_login"] else None
    }

# Form management endpoints
@app.post("/forms")
async def create_form(form_data: FormCreate, current_user: dict = Depends(get_current_user)):

    # Validate required fields
    if not form_data.title or not form_data.title.strip():
        raise HTTPException(status_code=400, detail="Form title is required")

    if not form_data.department_id:
        raise HTTPException(status_code=400, detail="Department ID is required")

    if not form_data.form_data:
        raise HTTPException(status_code=400, detail="Form data is required")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Parse and validate due_date if provided
                due_date_value = None
                if form_data.due_date:
                    try:
                        due_date_value = datetime.fromisoformat(form_data.due_date.replace('Z', '+00:00'))

                        # Validate due date is at least 24 hours in the future (skip for admin publishing)
                        if form_data.status != 'published' or current_user["role"] != "admin":
                            now = datetime.now(due_date_value.tzinfo) if due_date_value.tzinfo else datetime.now()
                            twenty_four_hours_from_now = now + timedelta(hours=24)

                            if due_date_value < twenty_four_hours_from_now:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Due date must be at least 24 hours in the future"
                                )
                    except ValueError:
                        raise HTTPException(status_code=400, detail="Invalid due date format")
                    except HTTPException:
                        raise  # Re-raise HTTP exceptions
                    except:
                        due_date_value = None

                # Insert form (using creator_id and trainer_id properly)

                # Set published_at if status is 'published'
                published_at_value = datetime.now() if form_data.status == 'published' else None

                cur.execute("""
                    INSERT INTO forms (title, description, creator_id, trainer_id, department_id, form_data, due_date, status, published_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id, title, description, creator_id, trainer_id, department_id, form_data, due_date, status, published_at, created_at, updated_at
                """, (form_data.title, form_data.description, current_user["id"], form_data.trainer_id,
                     form_data.department_id, json.dumps(form_data.form_data), due_date_value, form_data.status, published_at_value))

                form = dict(cur.fetchone())

                # Explicitly commit the transaction
                conn.commit()

                # Verify the form was actually saved
                cur.execute("SELECT COUNT(*) as count FROM forms WHERE id = %s", (form['id'],))
                verification = cur.fetchone()

                return {
                    "success": True,
                    "message": "Form created successfully",
                    "form": {
                        "id": form["id"],
                        "title": form["title"],
                        "description": form["description"],
                        "creator_id": form["creator_id"],
                        "department_id": form["department_id"],
                        "trainer_id": form["trainer_id"],
                        "form_data": form["form_data"],
                        "status": form["status"],
                        "due_date": form["due_date"].isoformat() if form["due_date"] else None,
                        "published_at": form["published_at"].isoformat() if form["published_at"] else None,
                        "is_published": form["status"] == "published",  # ✅ Add is_published field
                        "type": "single-use",
                        "created_at": form["created_at"].isoformat() if form["created_at"] else None,
                        "updated_at": form["updated_at"].isoformat() if form["updated_at"] else None
                    }
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Form creation failed: {str(e)}")

@app.get("/forms")
async def get_forms(current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if current_user["role"] == "admin":
                    # Admin sees all forms with response counts - using subquery for accurate counting
                    cur.execute("""
                        SELECT f.*, u.first_name || ' ' || u.last_name as creator_name,
                               d.name as department_name,
                               COALESCE(rc.response_count, 0) as response_count,
                               CASE WHEN f.status = 'published' THEN true ELSE false END as is_published
                        FROM forms f
                        JOIN users u ON f.creator_id = u.id
                        JOIN departments d ON f.department_id = d.id
                        LEFT JOIN (
                            SELECT form_id, COUNT(*) as response_count
                            FROM form_responses
                            GROUP BY form_id
                        ) rc ON f.id = rc.form_id
                        ORDER BY f.created_at DESC
                    """)
                elif current_user["role"] == "trainer":
                    # Trainer sees their own forms with response counts - using subquery for accurate counting
                    cur.execute("""
                        SELECT f.*, u.first_name || ' ' || u.last_name as creator_name,
                               d.name as department_name,
                               COALESCE(rc.response_count, 0) as response_count,
                               CASE WHEN f.status = 'published' THEN true ELSE false END as is_published
                        FROM forms f
                        JOIN users u ON f.creator_id = u.id
                        JOIN departments d ON f.department_id = d.id
                        LEFT JOIN (
                            SELECT form_id, COUNT(*) as response_count
                            FROM form_responses
                            GROUP BY form_id
                        ) rc ON f.id = rc.form_id
                        WHERE f.trainer_id = %s OR f.creator_id = %s
                        ORDER BY f.created_at DESC
                    """, (current_user["id"], current_user["id"]))
                else:
                    # Trainee sees forms for their department with completion status
                    cur.execute("""
                        SELECT f.*, u.first_name || ' ' || u.last_name as creator_name,
                               d.name as department_name,
                               CASE WHEN fr.id IS NOT NULL THEN true ELSE false END as is_completed,
                               CASE WHEN f.status = 'published' THEN true ELSE false END as is_published
                        FROM forms f
                        JOIN users u ON f.creator_id = u.id
                        JOIN departments d ON f.department_id = d.id
                        LEFT JOIN form_responses fr ON f.id = fr.form_id AND fr.user_id = %s
                        WHERE f.department_id = %s AND f.status = 'published'
                        ORDER BY f.created_at DESC
                    """, (current_user["id"], current_user["department_id"]))

                forms = []
                for row in cur.fetchall():
                    form_dict = dict(row)

                    # Parse form_data JSON if it exists
                    if form_dict.get("form_data"):
                        try:
                            if isinstance(form_dict["form_data"], str):
                                form_dict["form_data"] = json.loads(form_dict["form_data"])
                        except (json.JSONDecodeError, TypeError) as e:

                            form_dict["form_data"] = {"questions": [], "settings": {}}

                    # Convert to expected format
                    form_dict["creator_id"] = form_dict.get("creator_id") or form_dict.get("trainer_id")
                    form_dict["status"] = form_dict.get("status", "draft")
                    form_dict["type"] = "single-use"
                    form_dict["response_count"] = form_dict.get("response_count", 0)
                    form_dict["created_at"] = form_dict["created_at"].isoformat() if form_dict["created_at"] else None
                    form_dict["updated_at"] = form_dict["updated_at"].isoformat() if form_dict["updated_at"] else None
                    form_dict["published_at"] = form_dict["published_at"].isoformat() if form_dict["published_at"] else None
                    form_dict["due_date"] = form_dict["due_date"].isoformat() if form_dict.get("due_date") else None
                    forms.append(form_dict)

                return {
                    "success": True,
                    "forms": forms
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get forms: {str(e)}")

@app.get("/forms/{form_id}")
async def get_form(form_id: int, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT f.*,
                           COALESCE(u.first_name || ' ' || u.last_name, 'Unknown Trainer') as creator_name,
                           COALESCE(d.name, 'Unknown Department') as department_name,
                           CASE WHEN f.status = 'published' THEN true ELSE COALESCE(f.is_published, false) END as is_published
                    FROM forms f
                    LEFT JOIN users u ON COALESCE(f.creator_id, f.trainer_id) = u.id
                    LEFT JOIN departments d ON f.department_id = d.id
                    WHERE f.id = %s
                """, (form_id,))

                form = cur.fetchone()
                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                form_dict = dict(form)

                # Check permissions
                if (current_user["role"] == "trainee" and
                    (form_dict["department_id"] != current_user["department_id"] or
                     not form_dict.get("is_published", False))):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Parse form_data JSON if it exists
                if form_dict.get("form_data"):
                    try:
                        if isinstance(form_dict["form_data"], str):
                            form_dict["form_data"] = json.loads(form_dict["form_data"])
                    except (json.JSONDecodeError, TypeError) as e:

                        form_dict["form_data"] = {"questions": [], "settings": {}}

                # Convert to expected format
                form_dict["creator_id"] = form_dict["trainer_id"]
                form_dict["status"] = "published" if form_dict.get("is_published") else "draft"
                form_dict["type"] = "single-use"
                form_dict["created_at"] = form_dict["created_at"].isoformat() if form_dict["created_at"] else None
                form_dict["updated_at"] = form_dict["updated_at"].isoformat() if form_dict["updated_at"] else None
                form_dict["published_at"] = form_dict["published_at"].isoformat() if form_dict["published_at"] else None

                return {
                    "success": True,
                    "form": form_dict
                }
    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get form: {str(e)}")

@app.put("/forms/{form_id}")
async def update_form(form_id: int, form_data: dict, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists and user has permission
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                existing_form = cur.fetchone()

                if not existing_form:
                    raise HTTPException(status_code=404, detail="Form not found")

                existing_form_dict = dict(existing_form)

                # Check permissions (only creator or admin can update)
                if (current_user["role"] != "admin" and
                    existing_form_dict["creator_id"] != current_user["id"] and
                    existing_form_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Build update query dynamically
                update_fields = []
                update_values = []

                if "title" in form_data:
                    update_fields.append("title = %s")
                    update_values.append(form_data["title"])

                if "description" in form_data:
                    update_fields.append("description = %s")
                    update_values.append(form_data["description"])

                if "form_data" in form_data:
                    update_fields.append("form_data = %s")
                    update_values.append(json.dumps(form_data["form_data"]))

                if "due_date" in form_data:
                    due_date_value = None
                    if form_data["due_date"]:
                        try:
                            due_date_value = datetime.fromisoformat(form_data["due_date"].replace('Z', '+00:00'))

                            # Validate due date is at least 24 hours in the future
                            now = datetime.now(due_date_value.tzinfo) if due_date_value.tzinfo else datetime.now()
                            twenty_four_hours_from_now = now + timedelta(hours=24)

                            if due_date_value < twenty_four_hours_from_now:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Due date must be at least 24 hours in the future"
                                )
                        except ValueError:
                            raise HTTPException(status_code=400, detail="Invalid due date format")
                        except HTTPException:
                            raise  # Re-raise HTTP exceptions

                    update_fields.append("due_date = %s")
                    update_values.append(due_date_value)

                if "status" in form_data:
                    is_published = form_data["status"] == "published"
                    update_fields.append("is_published = %s")
                    update_values.append(is_published)

                    if is_published:
                        update_fields.append("published_at = CURRENT_TIMESTAMP")

                update_fields.append("updated_at = CURRENT_TIMESTAMP")
                update_values.append(form_id)

                if not update_fields:
                    raise HTTPException(status_code=400, detail="No fields to update")

                # Execute update
                query = f"UPDATE forms SET {', '.join(update_fields)} WHERE id = %s RETURNING *"
                cur.execute(query, update_values)

                updated_form = dict(cur.fetchone())
                conn.commit()

                # Convert to expected format
                updated_form["creator_id"] = updated_form["trainer_id"]
                updated_form["status"] = "published" if updated_form.get("is_published") else "draft"
                updated_form["type"] = "single-use"
                updated_form["created_at"] = updated_form["created_at"].isoformat() if updated_form["created_at"] else None
                updated_form["updated_at"] = updated_form["updated_at"].isoformat() if updated_form["updated_at"] else None
                updated_form["published_at"] = updated_form["published_at"].isoformat() if updated_form["published_at"] else None

                return {
                    "success": True,
                    "message": "Form updated successfully",
                    "form": updated_form
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to update form: {str(e)}")

@app.put("/forms/{form_id}/publish")
async def publish_form(form_id: int, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists and user has permission
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                existing_form = cur.fetchone()

                if not existing_form:
                    raise HTTPException(status_code=404, detail="Form not found")

                existing_form_dict = dict(existing_form)

                # Check permissions (only creator or admin can publish)
                if (current_user["role"] != "admin" and
                    existing_form_dict["creator_id"] != current_user["id"] and
                    existing_form_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Update form to published
                cur.execute("""
                    UPDATE forms
                    SET status = 'published', published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING *
                """, (form_id,))

                published_form = dict(cur.fetchone())
                conn.commit()

                # Convert to expected format
                published_form["creator_id"] = published_form["trainer_id"]
                published_form["status"] = "published"
                published_form["type"] = "single-use"
                published_form["created_at"] = published_form["created_at"].isoformat() if published_form["created_at"] else None
                published_form["updated_at"] = published_form["updated_at"].isoformat() if published_form["updated_at"] else None
                published_form["published_at"] = published_form["published_at"].isoformat() if published_form["published_at"] else None

                return {
                    "success": True,
                    "message": "Form published successfully",
                    "form": published_form
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to publish form: {str(e)}")

@app.put("/forms/{form_id}/save-draft")
async def save_form_draft(form_id: int, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists and user has permission
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                existing_form = cur.fetchone()

                if not existing_form:
                    raise HTTPException(status_code=404, detail="Form not found")

                existing_form_dict = dict(existing_form)

                # Check permissions (only creator or admin can save as draft)
                if (current_user["role"] != "admin" and
                    existing_form_dict["creator_id"] != current_user["id"] and
                    existing_form_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Update form to draft status
                cur.execute("""
                    UPDATE forms
                    SET status = 'draft', updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING *
                """, (form_id,))

                draft_form = dict(cur.fetchone())
                conn.commit()

                # Convert to expected format
                draft_form["creator_id"] = draft_form.get("creator_id") or draft_form["trainer_id"]
                draft_form["status"] = "draft"
                draft_form["type"] = "single-use"
                draft_form["created_at"] = draft_form["created_at"].isoformat() if draft_form["created_at"] else None
                draft_form["updated_at"] = draft_form["updated_at"].isoformat() if draft_form["updated_at"] else None
                draft_form["published_at"] = draft_form["published_at"].isoformat() if draft_form["published_at"] else None
                draft_form["due_date"] = draft_form["due_date"].isoformat() if draft_form.get("due_date") else None

                return {
                    "success": True,
                    "message": "Form saved as draft successfully",
                    "form": draft_form
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to save form as draft: {str(e)}")

# Delete form endpoint
@app.delete("/forms/{form_id}")
async def delete_form(form_id: int, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                form_dict = dict(form)

                # Check permissions (only creator or admin can delete)
                if (current_user["role"] != "admin" and
                    form_dict["creator_id"] != current_user["id"] and
                    form_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Delete the form (CASCADE will handle related records)
                cur.execute("DELETE FROM forms WHERE id = %s", (form_id,))

                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Form not found or already deleted")

                # Explicitly commit the transaction
                conn.commit()

                # Verify the form was actually deleted
                cur.execute("SELECT COUNT(*) as count FROM forms WHERE id = %s", (form_id,))
                verification = cur.fetchone()

                return {
                    "success": True,
                    "message": "Form deleted successfully"
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to delete form: {str(e)}")

# Form deletion request endpoints
@app.post("/forms/{form_id}/deletion-request")
async def create_form_deletion_request(form_id: int, request_data: FormDeletionRequest, current_user: dict = Depends(get_current_user)):
    """Create a deletion request for a form (trainers only)"""

    # Only trainers can request form deletion
    if current_user["role"] != "trainer":
        raise HTTPException(
            status_code=403,
            detail="Access denied. Only trainers can request form deletion."
        )

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists
                cur.execute("""
                    SELECT f.*, u.first_name || ' ' || u.last_name as trainer_name
                    FROM forms f
                    LEFT JOIN users u ON f.trainer_id = u.id
                    WHERE f.id = %s
                """, (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(
                        status_code=404,
                        detail="Form not found. The specified form does not exist."
                    )

                form_dict = dict(form)

                # Check if trainer owns the form
                if form_dict["trainer_id"] != current_user["id"]:
                    trainer_name = form_dict.get("trainer_name", "another trainer")
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied. This form belongs to {trainer_name}. You can only request deletion of your own forms."
                    )

                # Check if form has any active responses
                cur.execute("""
                    SELECT COUNT(*) FROM form_responses WHERE form_id = %s
                """, (form_id,))
                response_count = cur.fetchone()["count"]

                if response_count > 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot request deletion. This form has {response_count} responses. Please contact an administrator for assistance."
                    )

                # Check if there's already any deletion request
                cur.execute("""
                    SELECT status FROM form_deletion_requests 
                    WHERE form_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (form_id,))
                existing_request = cur.fetchone()

                if existing_request:
                    status = existing_request["status"]
                    if status == "pending":
                        raise HTTPException(
                            status_code=400,
                            detail="A deletion request for this form is already pending review."
                        )
                    elif status == "approved":
                        raise HTTPException(
                            status_code=400,
                            detail="This form has already been approved for deletion."
                        )

                # Create deletion request with explicit status
                cur.execute("""
                    INSERT INTO form_deletion_requests 
                    (form_id, trainer_id, reason, status, created_at)
                    VALUES (%s, %s, %s, 'pending', CURRENT_TIMESTAMP)
                    RETURNING id, form_id, trainer_id, reason, status, created_at
                """, (form_id, current_user["id"], request_data.reason))

                deletion_request = dict(cur.fetchone())
                conn.commit()

                return {
                    "success": True,
                    "message": "Form deletion request submitted successfully and is pending admin review.",
                    "request": {
                        "id": deletion_request["id"],
                        "form_id": deletion_request["form_id"],
                        "trainer_id": deletion_request["trainer_id"],
                        "reason": deletion_request["reason"],
                        "status": deletion_request["status"],
                        "created_at": deletion_request["created_at"].isoformat() if deletion_request["created_at"] else None,
                        "form_title": form_dict["title"]
                    }
                }

    except HTTPException:
        raise
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create deletion request: {str(e)}"
        )

@app.get("/form-deletion-requests")
async def get_form_deletion_requests(current_user: dict = Depends(get_current_user)):
    """Get form deletion requests (admin: all, trainer: own requests)"""

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if current_user["role"] == "admin":
                    # Admin can see all deletion requests
                    cur.execute("""
                        SELECT fdr.*, f.title as form_title, f.description as form_description,
                               u.first_name, u.last_name, u.email as trainer_email
                        FROM form_deletion_requests fdr
                        JOIN forms f ON fdr.form_id = f.id
                        JOIN users u ON fdr.trainer_id = u.id
                        ORDER BY fdr.created_at DESC
                    """)
                else:
                    # Trainers can only see their own deletion requests
                    cur.execute("""
                        SELECT fdr.*, f.title as form_title, f.description as form_description
                        FROM form_deletion_requests fdr
                        JOIN forms f ON fdr.form_id = f.id
                        WHERE fdr.trainer_id = %s
                        ORDER BY fdr.created_at DESC
                    """, (current_user["id"],))

                requests = cur.fetchall()
                requests_list = []

                for req in requests:
                    req_dict = dict(req)
                    request_data = {
                        "id": req_dict["id"],
                        "form_id": req_dict["form_id"],
                        "trainer_id": req_dict["trainer_id"],
                        "reason": req_dict["reason"],
                        "status": req_dict["status"],
                        "admin_response": req_dict.get("admin_response"),
                        "reviewed_by": req_dict.get("reviewed_by"),
                        "reviewed_at": req_dict["reviewed_at"].isoformat() if req_dict.get("reviewed_at") else None,
                        "created_at": req_dict["created_at"].isoformat() if req_dict["created_at"] else None,
                        "form_title": req_dict["form_title"],
                        "form_description": req_dict.get("form_description")
                    }

                    # Add trainer info for admin view
                    if current_user["role"] == "admin":
                        request_data.update({
                            "trainer_name": f"{req_dict['first_name']} {req_dict['last_name']}",
                            "trainer_email": req_dict["trainer_email"]
                        })

                    requests_list.append(request_data)

                return {
                    "success": True,
                    "requests": requests_list
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get deletion requests: {str(e)}")

@app.put("/form-deletion-requests/{request_id}/approve")
async def approve_form_deletion_request(request_id: int, current_user: dict = Depends(get_current_user)):
    """Approve a form deletion request and delete the form (admin only)"""

    # Only admins can approve deletion requests
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied. Only administrators can approve deletion requests."
        )

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Start transaction
                conn.autocommit = False

                try:
                    # Get the deletion request and form details - removed FOR UPDATE to prevent deadlocks
                    cur.execute("""
                        SELECT fdr.*, f.title as form_title, f.trainer_id,
                               u.first_name || ' ' || u.last_name as trainer_name
                        FROM form_deletion_requests fdr
                        JOIN forms f ON fdr.form_id = f.id
                        JOIN users u ON f.trainer_id = u.id
                        WHERE fdr.id = %s
                    """, (request_id,))

                    deletion_request = cur.fetchone()
                    if not deletion_request:
                        raise HTTPException(
                            status_code=404,
                            detail="Deletion request not found. It may have been already processed or deleted."
                        )

                    req_dict = dict(deletion_request)
                    form_id = req_dict["form_id"]

                    # Verify request is still pending
                    if req_dict["status"] != "pending":
                        raise HTTPException(
                            status_code=400,
                            detail=f"Cannot approve request. Current status is '{req_dict['status']}'."
                        )

                    # Check for any dependencies or constraints
                    cur.execute("""
                        SELECT COUNT(*) as count FROM form_responses 
                        WHERE form_id = %s
                    """, (form_id,))
                    response_count = cur.fetchone()["count"]

                    if response_count > 0:
                        # Update request status to need_review instead of failing
                        cur.execute("""
                            UPDATE form_deletion_requests
                            SET status = 'need_review',
                                admin_response = %s,
                                reviewed_by = %s,
                                reviewed_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                            RETURNING *
                        """, (
                            f"Form has {response_count} responses. Manual review required.",
                            current_user["id"],
                            request_id
                        ))
                        
                        conn.commit()
                        
                        return {
                            "success": False,
                            "message": f"Form '{req_dict['form_title']}' has {response_count} responses and cannot be automatically deleted. Manual review required.",
                            "request": {
                                "id": req_dict["id"],
                                "form_id": req_dict["form_id"],
                                "status": "need_review",
                                "admin_response": f"Form has {response_count} responses. Manual review required."
                            }
                        }

                    # Begin deletion process
                    # 1. Delete form responses (if any remain)
                    cur.execute("DELETE FROM form_responses WHERE form_id = %s", (form_id,))
                    
                    # 2. Delete any other related data
                    # Add other cleanup queries here if needed
                    
                    # 3. Delete the form itself
                    cur.execute("DELETE FROM forms WHERE id = %s", (form_id,))
                    
                    if cur.rowcount == 0:
                        raise HTTPException(
                            status_code=404,
                            detail="Form not found or already deleted."
                        )

                    # 4. Update the deletion request status
                    cur.execute("""
                        UPDATE form_deletion_requests
                        SET status = 'approved',
                            admin_response = 'Form and related data deleted successfully',
                            reviewed_by = %s,
                            reviewed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING *
                    """, (current_user["id"], request_id))

                    updated_request = dict(cur.fetchone())
                    
                    # If we get here, everything succeeded, so commit the transaction
                    conn.commit()

                    return {
                        "success": True,
                        "message": f"Form '{req_dict['form_title']}' has been successfully deleted",
                        "request": {
                            "id": updated_request["id"],
                            "form_id": updated_request["form_id"],
                            "status": updated_request["status"],
                            "admin_response": updated_request["admin_response"],
                            "reviewed_at": updated_request["reviewed_at"].isoformat() if updated_request["reviewed_at"] else None,
                            "trainer_name": req_dict["trainer_name"]
                        }
                    }

                except HTTPException as http_error:
                    conn.rollback()
                    raise http_error
                except psycopg2.Error as db_error:
                    conn.rollback()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Database error: {str(db_error)}"
                    )
                except Exception as e:
                    conn.rollback()
                    raise e
                finally:
                    conn.autocommit = True

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process deletion request: {str(e)}"
        )

@app.put("/form-deletion-requests/{request_id}/reject")
async def reject_form_deletion_request(request_id: int, rejection_data: dict, current_user: dict = Depends(get_current_user)):
    """Reject a form deletion request (admin only)"""

    # Only admins can reject deletion requests
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can reject deletion requests")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the deletion request
                cur.execute("""
                    SELECT fdr.*, f.title as form_title
                    FROM form_deletion_requests fdr
                    JOIN forms f ON fdr.form_id = f.id
                    WHERE fdr.id = %s AND fdr.status = 'pending'
                """, (request_id,))

                deletion_request = cur.fetchone()
                if not deletion_request:
                    raise HTTPException(status_code=404, detail="Deletion request not found or already processed")

                req_dict = dict(deletion_request)
                reason = rejection_data.get("reason", "Request rejected by admin")

                # Update the deletion request status
                cur.execute("""
                    UPDATE form_deletion_requests
                    SET status = 'rejected',
                        admin_response = %s,
                        reviewed_by = %s,
                        reviewed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING *
                """, (reason, current_user["id"], request_id))

                updated_request = dict(cur.fetchone())
                conn.commit()

                return {
                    "success": True,
                    "message": f"Deletion request for '{req_dict['form_title']}' has been rejected",
                    "request": {
                        "id": updated_request["id"],
                        "form_id": updated_request["form_id"],
                        "status": updated_request["status"],
                        "admin_response": updated_request["admin_response"],
                        "reviewed_at": updated_request["reviewed_at"].isoformat() if updated_request["reviewed_at"] else None
                    }
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to reject deletion request: {str(e)}")

# Trainers endpoint
@app.get("/trainers")
async def get_trainers():
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT u.*, d.name as department_name
                    FROM users u
                    LEFT JOIN departments d ON u.department_id = d.id
                    WHERE u.role = 'trainer'
                    ORDER BY u.first_name, u.last_name
                """)
                trainers = []
                for row in cur.fetchall():
                    trainer_dict = dict(row)
                    trainer_dict["created_at"] = trainer_dict["created_at"].isoformat() if trainer_dict["created_at"] else None
                    trainer_dict["updated_at"] = trainer_dict["updated_at"].isoformat() if trainer_dict["updated_at"] else None
                    trainers.append(trainer_dict)

                return {
                    "success": True,
                    "trainers": trainers
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trainers: {str(e)}")

# Form templates endpoint
@app.get("/form-templates/default")
async def get_default_form_template(request_id: Optional[int] = None):
    try:
        # Base template with essential questions only
        base_questions = [
            {
                "id": "session_rating",
                "type": "rating",
                "question": "How would you rate this training session overall?",
                "required": True,
                "options": {"max": 5, "min": 1}
            },
            {
                "id": "content_clarity",
                "type": "rating",
                "question": "How clear was the content presented?",
                "required": True,
                "options": {"max": 5, "min": 1}
            },
            {
                "id": "what_learned",
                "type": "textarea",
                "question": "What did you learn from this session?",
                "required": True
            },
            {
                "id": "improvements",
                "type": "textarea",
                "question": "What could be improved?",
                "required": False
            }
        ]

        # If request_id is provided, customize template based on form request
        if request_id:
            with get_db() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM form_requests WHERE id = %s", (request_id,))
                    request_data = cur.fetchone()

                    if request_data:
                        request_dict = dict(request_data)

                        # If the request has complete form_data (from trainer), use it
                        if request_dict.get("form_data"):
                            try:
                                form_data = json.loads(request_dict["form_data"]) if isinstance(request_dict["form_data"], str) else request_dict["form_data"]
                                if form_data.get("questions"):
                                    base_questions = form_data["questions"]

                            except (json.JSONDecodeError, TypeError) as e:
                                pass  # Use default base_questions if form_data parsing fails

                        # Customize template based on request type and additional notes (fallback)
                        elif request_dict.get("form_type") == "evaluation":
                            base_questions.append({
                                "id": "instructor_performance",
                                "type": "rating",
                                "question": "How would you rate the instructor's performance?",
                                "required": True,
                                "options": {"max": 5, "min": 1}
                            })

                        # Add custom questions based on additional notes (fallback)
                        elif request_dict.get("additional_notes"):
                            base_questions.append({
                                "id": "custom_feedback",
                                "type": "textarea",
                                "question": f"Additional feedback: {request_dict['additional_notes']}",
                                "required": False
                            })

        default_template = {
            "title": "Training Session Feedback Form",
            "description": "Please provide your feedback on the training session",
            "form_data": {
                "questions": base_questions,
                "settings": {},
                "session": {
                    "name": "",
                    "date": "",
                    "course": "",
                    "duration": 60
                }
            }
        }

        return {
            "success": True,
            "template": default_template
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get default template: {str(e)}")

@app.get("/trainer/form-template")
async def get_trainer_form_template(current_user: dict = Depends(get_current_user)):
    """Get default form template for trainers to use in form builder"""
    if current_user["role"] not in ["trainer", "admin"]:
        raise HTTPException(status_code=403, detail="Trainer or admin access required")

    try:
        # Return a basic template that trainers can use
        template = {
            "title": "",
            "description": "",
            "form_data": {
                "questions": [
                    {
                        "id": "overall_rating",
                        "type": "rating",
                        "question": "How would you rate this training session overall?",
                        "required": True,
                        "options": {"max": 5, "min": 1, "labels": ["Poor", "Fair", "Good", "Very Good", "Excellent"]}
                    },
                    {
                        "id": "content_quality",
                        "type": "rating",
                        "question": "How would you rate the quality of the training content?",
                        "required": True,
                        "options": {"max": 5, "min": 1}
                    },
                    {
                        "id": "instructor_effectiveness",
                        "type": "rating",
                        "question": "How effective was the instructor?",
                        "required": True,
                        "options": {"max": 5, "min": 1}
                    },
                    {
                        "id": "additional_feedback",
                        "type": "textarea",
                        "question": "Please provide any additional feedback or suggestions:",
                        "required": False,
                        "placeholder": "Your feedback helps us improve our training programs..."
                    }
                ],
                "settings": {
                    "allowAnonymous": True,
                    "requireAll": False,
                    "showProgress": True,
                    "randomizeQuestions": False
                }
            }
        }

        return {
            "success": True,
            "template": template
        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get trainer form template: {str(e)}")

# Form requests endpoints
@app.get("/form-requests")
async def get_form_requests(current_user: dict = Depends(get_current_user)):
    try:

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get form requests based on user role
                if current_user["role"] == "admin":
                    # Admins see all pending requests (not yet converted to forms and not rejected)
                    cur.execute("""
                        SELECT
                            fr.*,
                            u.first_name || ' ' || u.last_name as trainer_name,
                            u.email as trainer_email,
                            d.name as department_name
                        FROM form_requests fr
                        LEFT JOIN users u ON fr.trainer_id = u.id
                        LEFT JOIN departments d ON fr.department_id = d.id
                        WHERE fr.form_created IS NOT TRUE AND (fr.status IS NULL OR fr.status != 'rejected')
                        ORDER BY fr.created_at DESC
                    """)
                elif current_user["role"] == "trainer":
                    # Trainers see their own requests (pending and rejected, but not converted to forms)
                    cur.execute("""
                        SELECT
                            fr.*,
                            u.first_name || ' ' || u.last_name as trainer_name,
                            u.email as trainer_email,
                            d.name as department_name
                        FROM form_requests fr
                        LEFT JOIN users u ON fr.trainer_id = u.id
                        LEFT JOIN departments d ON fr.department_id = d.id
                        WHERE fr.trainer_id = %s AND fr.form_created IS NOT TRUE
                        ORDER BY fr.created_at DESC
                    """, (current_user["id"],))
                else:
                    # Other roles don't have access to form requests
                    raise HTTPException(status_code=403, detail="Access denied - insufficient permissions")

                requests = cur.fetchall()

                # Convert to list of dictionaries and debug form_data parsing
                requests_list = []
                for request in requests:
                    request_dict = dict(request)

                    # Debug form_data parsing
                    if request_dict.get('form_data'):
                        try:
                            if isinstance(request_dict['form_data'], str):
                                parsed_data = json.loads(request_dict['form_data'])
                                request_dict['form_data'] = parsed_data

                            else:
                                # form_data is already parsed
                                pass

                        except json.JSONDecodeError as e:
                            # Keep original string if parsing fails
                            pass

                    else:
                        # No form_data to parse
                        pass

                    requests_list.append(request_dict)

                return {
                    "success": True,
                    "requests": requests_list
                }
    except Exception as e:
        print(f"Error getting form requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get form requests: {str(e)}")

@app.get("/form-requests/trainer/{trainer_id}")
async def get_trainer_form_requests(trainer_id: int, current_user: dict = Depends(get_current_user)):
    """Get all form requests for a specific trainer"""

    # Check permissions - trainers can only see their own requests, admins can see all
    if current_user["role"] != "admin" and current_user["id"] != trainer_id:
        raise HTTPException(status_code=403, detail="You can only view your own form requests")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all form requests for the trainer
                cur.execute("""
                    SELECT fr.*, u.first_name, u.last_name, u.email, d.name as department_name
                    FROM form_requests fr
                    JOIN users u ON fr.trainer_id = u.id
                    LEFT JOIN departments d ON fr.department_id = d.id
                    WHERE fr.trainer_id = %s
                    ORDER BY fr.created_at DESC
                """, (trainer_id,))

                requests = cur.fetchall()
                requests_list = []

                for req in requests:
                    req_dict = dict(req)

                    # Parse form_data if it's a string
                    form_data = req_dict.get("form_data")
                    if isinstance(form_data, str):
                        try:
                            form_data = json.loads(form_data)
                        except (json.JSONDecodeError, TypeError):
                            form_data = None

                    request_dict = {
                        "id": req_dict["id"],
                        "trainer_id": req_dict["trainer_id"],
                        "title": req_dict["title"],
                        "description": req_dict["description"],
                        "department_id": req_dict["department_id"],
                        "department_name": req_dict["department_name"],
                        "session_name": req_dict["session_name"],
                        "session_date": req_dict["session_date"].isoformat() if req_dict["session_date"] else None,
                        "session_duration": req_dict["session_duration"],
                        "form_validity_duration": req_dict["form_validity_duration"],
                        "form_type": req_dict["form_type"],
                        "priority": req_dict["priority"],
                        "additional_notes": req_dict["additional_notes"],
                        "status": req_dict["status"],
                        "admin_response": req_dict["admin_response"],
                        "reviewed_by": req_dict["reviewed_by"],
                        "reviewed_at": req_dict["reviewed_at"].isoformat() if req_dict["reviewed_at"] else None,
                        "created_at": req_dict["created_at"].isoformat() if req_dict["created_at"] else None,
                        "due_date": req_dict.get("due_date"),
                        "form_data": form_data,
                        "trainer_name": f"{req_dict['first_name']} {req_dict['last_name']}",
                        "trainer_email": req_dict["email"]
                    }

                    requests_list.append(request_dict)

                return {
                    "success": True,
                    "requests": requests_list
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get trainer form requests: {str(e)}")

@app.get("/form-requests/rejected/{trainer_id}")
async def get_rejected_form_requests(trainer_id: int, current_user: dict = Depends(get_current_user)):
    """Get rejected form requests for a specific trainer"""

    # Check permissions - trainers can only see their own requests, admins can see all
    if current_user["role"] != "admin" and current_user["id"] != trainer_id:
        raise HTTPException(status_code=403, detail="You can only view your own form requests")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get rejected form requests for the trainer
                cur.execute("""
                    SELECT fr.*, u.first_name, u.last_name, u.email, d.name as department_name
                    FROM form_requests fr
                    JOIN users u ON fr.trainer_id = u.id
                    LEFT JOIN departments d ON fr.department_id = d.id
                    WHERE fr.trainer_id = %s AND fr.status = 'rejected'
                    ORDER BY fr.reviewed_at DESC, fr.created_at DESC
                """, (trainer_id,))

                requests = cur.fetchall()
                requests_list = []

                for req in requests:
                    req_dict = dict(req)

                    # Parse form_data if it's a string
                    form_data = req_dict.get("form_data")
                    if isinstance(form_data, str):
                        try:
                            form_data = json.loads(form_data)
                        except (json.JSONDecodeError, TypeError):
                            form_data = None

                    request_dict = {
                        "id": req_dict["id"],
                        "trainer_id": req_dict["trainer_id"],
                        "title": req_dict["title"],
                        "description": req_dict["description"],
                        "department_id": req_dict["department_id"],
                        "department_name": req_dict["department_name"],
                        "session_name": req_dict["session_name"],
                        "session_date": req_dict["session_date"].isoformat() if req_dict["session_date"] else None,
                        "session_duration": req_dict["session_duration"],
                        "form_validity_duration": req_dict["form_validity_duration"],
                        "form_type": req_dict["form_type"],
                        "priority": req_dict["priority"],
                        "additional_notes": req_dict["additional_notes"],
                        "status": req_dict["status"],
                        "admin_response": req_dict["admin_response"],
                        "reviewed_by": req_dict["reviewed_by"],
                        "reviewed_at": req_dict["reviewed_at"].isoformat() if req_dict["reviewed_at"] else None,
                        "created_at": req_dict["created_at"].isoformat() if req_dict["created_at"] else None,
                        "due_date": req_dict.get("due_date"),
                        "form_data": form_data,
                        "trainer_name": f"{req_dict['first_name']} {req_dict['last_name']}",
                        "trainer_email": req_dict["email"]
                    }

                    requests_list.append(request_dict)

                return {
                    "success": True,
                    "requests": requests_list
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get rejected form requests: {str(e)}")

@app.post("/form-requests")
async def create_form_request(request_data: dict, current_user: dict = Depends(get_current_user)):

    # Temporarily removed permission check to test database insertion

    try:
        # Save form request to database
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Insert form request into database
                cur.execute("""
                    INSERT INTO form_requests (
                        title, description, trainer_id, department_id,
                        session_name, session_date, session_duration,
                        form_validity_duration, form_type, priority,
                        additional_notes, status, created_at, form_data, due_date
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING *
                """, (
                    request_data.get("title", ""),
                    request_data.get("description", ""),
                    current_user["id"],
                    request_data.get("department_id", 1),
                    request_data.get("session_name", ""),
                    request_data.get("session_date", ""),
                    request_data.get("session_duration", 60),
                    request_data.get("form_validity_duration", 7),
                    request_data.get("form_type", "feedback"),
                    request_data.get("priority", "normal"),
                    request_data.get("additional_notes", ""),
                    "pending",
                    datetime.utcnow(),
                    json.dumps(request_data.get("form_data", {})) if request_data.get("form_data") else None,
                    request_data.get("due_date")
                ))

                new_request = cur.fetchone()
                conn.commit()

                return {
                    "success": True,
                    "message": "Form request submitted successfully",
                    "request": dict(new_request)
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to create form request: {str(e)}")

@app.put("/form-requests/{request_id}/approve")
async def approve_form_request(request_id: int, current_user: dict = Depends(get_current_user)):

    # Check if user is an admin (only admins can approve form requests)
    if current_user["role"] != "admin":

        raise HTTPException(status_code=403, detail="Only admins can approve form requests")

    try:

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Update form request status to approved
                cur.execute("""
                    UPDATE form_requests
                    SET status = 'approved',
                        reviewed_by = %s,
                        reviewed_at = %s,
                        admin_response = 'Request approved'
                    WHERE id = %s
                    RETURNING *
                """, (current_user["id"], datetime.utcnow(), request_id))

                updated_request = cur.fetchone()
                if not updated_request:
                    raise HTTPException(status_code=404, detail="Form request not found")

                conn.commit()

                return {
                    "success": True,
                    "message": "Form request approved successfully",
                    "request": dict(updated_request)
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to approve form request: {str(e)}")

@app.put("/form-requests/{request_id}/processing")
async def mark_request_processing(request_id: int, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Update request status to processing
                cur.execute("""
                    UPDATE form_requests
                    SET status = 'processing',
                        reviewed_by = %s,
                        reviewed_at = %s
                    WHERE id = %s
                    RETURNING *
                """, (current_user["id"], datetime.utcnow(), request_id))

                updated_request = cur.fetchone()

                if not updated_request:
                    raise HTTPException(status_code=404, detail="Form request not found")

                conn.commit()

                return {
                    "success": True,
                    "message": "Form request marked as processing",
                    "request": dict(updated_request)
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to mark request as processing: {str(e)}")

@app.put("/form-requests/{request_id}/reject")
async def reject_form_request(request_id: int, rejection_data: dict, current_user: dict = Depends(get_current_user)):

    # Check if user is an admin (only admins can reject form requests)
    if current_user["role"] != "admin":

        raise HTTPException(status_code=403, detail="Only admins can reject form requests")

    try:

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Update form request status to rejected
                cur.execute("""
                    UPDATE form_requests
                    SET status = 'rejected',
                        reviewed_by = %s,
                        reviewed_at = %s,
                        admin_response = %s
                    WHERE id = %s
                    RETURNING *
                """, (
                    current_user["id"],
                    datetime.utcnow(),
                    rejection_data.get("reason", "Request rejected"),
                    request_id
                ))

                updated_request = cur.fetchone()
                if not updated_request:
                    raise HTTPException(status_code=404, detail="Form request not found")

                conn.commit()

                return {
                    "success": True,
                    "message": "Form request rejected successfully",
                    "request": dict(updated_request)
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to reject form request: {str(e)}")

# Delete form request endpoint (for trainers to delete their own rejected requests)
@app.delete("/form-requests/{request_id}")
async def delete_form_request(request_id: int, current_user: dict = Depends(get_current_user)):

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if request exists and get its details
                cur.execute("SELECT * FROM form_requests WHERE id = %s", (request_id,))
                request = cur.fetchone()

                if not request:
                    raise HTTPException(status_code=404, detail="Form request not found")

                request_dict = dict(request)

                # Check permissions - trainers can only delete their own requests, admins can delete any
                if (current_user["role"] != "admin" and
                    request_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="You can only delete your own form requests")

                # Only allow deletion of rejected requests (to prevent accidental deletion of pending/approved ones)
                if request_dict["status"] not in ["rejected"]:
                    raise HTTPException(status_code=400, detail="Only rejected form requests can be deleted")

                # Delete the form request
                cur.execute("DELETE FROM form_requests WHERE id = %s", (request_id,))

                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Form request not found or already deleted")

                conn.commit()

                return {
                    "success": True,
                    "message": "Form request deleted successfully"
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to delete form request: {str(e)}")

@app.put("/form-requests/{request_id}/mark-processed")
async def mark_form_request_processed(request_id: int, current_user: dict = Depends(get_current_user)):

    # Check if user is an admin (only admins can mark requests as processed)
    if current_user["role"] != "admin":

        raise HTTPException(status_code=403, detail="Only admins can mark form requests as processed")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Update form request status to processed
                cur.execute("""
                    UPDATE form_requests
                    SET status = 'processed',
                        reviewed_by = %s,
                        reviewed_at = %s,
                        form_created = TRUE
                    WHERE id = %s
                    RETURNING *
                """, (
                    current_user["id"],
                    datetime.utcnow(),
                    request_id
                ))

                updated_request = cur.fetchone()
                if not updated_request:
                    raise HTTPException(status_code=404, detail="Form request not found")

                conn.commit()

                return {
                    "success": True,
                    "message": "Form request marked as processed successfully",
                    "request": dict(updated_request)
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to mark form request as processed: {str(e)}")

@app.post("/form-requests/{request_id}/create-form")
async def create_form_from_request(request_id: int, current_user: dict = Depends(get_current_user)):
    """Create a form from an approved form request"""

    # Check if user is an admin (only admins can create forms from requests)
    if current_user["role"] != "admin":

        raise HTTPException(status_code=403, detail="Only admins can create forms from requests")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the form request details
                cur.execute("""
                    SELECT fr.*, u.first_name, u.last_name, u.email
                    FROM form_requests fr
                    JOIN users u ON fr.trainer_id = u.id
                    WHERE fr.id = %s
                """, (request_id,))

                request_data = cur.fetchone()
                if not request_data:
                    raise HTTPException(status_code=404, detail="Form request not found")

                request_dict = dict(request_data)

                # Check if request is approved
                if request_dict["status"] != "approved":
                    raise HTTPException(status_code=400, detail="Form request must be approved before creating a form")

                # Check if form already exists for this request
                cur.execute("SELECT id FROM forms WHERE form_request_id = %s", (request_id,))
                existing_form = cur.fetchone()
                if existing_form:
                    raise HTTPException(status_code=400, detail="Form already exists for this request")

                # Generate default form structure based on request
                default_form_data = {
                    "questions": [
                        {
                            "id": 1,
                            "type": "rating",
                            "title": "How would you rate the overall quality of this training?",
                            "required": True,
                            "scale": {"min": 1, "max": 5, "labels": ["Poor", "Excellent"]}
                        },
                        {
                            "id": 2,
                            "type": "rating",
                            "title": "How clear were the explanations provided?",
                            "required": True,
                            "scale": {"min": 1, "max": 5, "labels": ["Very Unclear", "Very Clear"]}
                        },
                        {
                            "id": 3,
                            "type": "text",
                            "title": "What did you like most about this training?",
                            "required": False
                        },
                        {
                            "id": 4,
                            "type": "text",
                            "title": "What could be improved?",
                            "required": False
                        }
                    ],
                    "settings": {
                        "allowAnonymous": True,
                        "requireAll": False
                    }
                }

                # Parse due date from request or set default
                due_date_value = None
                if request_dict.get("due_date"):
                    try:
                        due_date_value = datetime.fromisoformat(str(request_dict["due_date"]).replace('Z', '+00:00'))
                    except:
                        # Default to 2 weeks from now
                        due_date_value = datetime.now() + timedelta(days=14)
                else:
                    # Default to 2 weeks from now
                    due_date_value = datetime.now() + timedelta(days=14)

                # Prepare form data - use trainer's form data if available, otherwise use default
                if request_dict.get("form_data"):
                    # Use the trainer's form data
                    form_data_to_use = request_dict["form_data"]

                else:
                    # Fallback to default template
                    form_data_to_use = default_form_data

                # Create the form
                cur.execute("""
                    INSERT INTO forms (
                        title, description, creator_id, trainer_id, department_id,
                        form_data, due_date, status, form_request_id, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id, title, description, creator_id, trainer_id, department_id,
                             form_data, due_date, status, form_request_id, created_at, updated_at
                """, (
                    request_dict["title"],
                    request_dict["description"],
                    current_user["id"],  # Admin who created the form
                    request_dict["trainer_id"],  # Trainer from the request
                    request_dict["department_id"],
                    json.dumps(form_data_to_use),  # ✅ Use trainer's form data instead of default
                    due_date_value,
                    'draft',  # Start as draft
                    request_id
                ))

                new_form = dict(cur.fetchone())

                # Update the form request to mark it as form_created
                cur.execute("""
                    UPDATE form_requests
                    SET form_created = true,
                        form_created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (request_id,))

                conn.commit()

                return {
                    "success": True,
                    "message": f"Form created successfully from request and assigned to {request_dict['first_name']} {request_dict['last_name']}",
                    "form": {
                        "id": new_form["id"],
                        "title": new_form["title"],
                        "description": new_form["description"],
                        "creator_id": new_form["creator_id"],
                        "trainer_id": new_form["trainer_id"],
                        "department_id": new_form["department_id"],
                        "form_data": new_form["form_data"],
                        "status": new_form["status"],
                        "due_date": new_form["due_date"].isoformat() if new_form["due_date"] else None,
                        "form_request_id": new_form["form_request_id"],
                        "created_at": new_form["created_at"].isoformat() if new_form["created_at"] else None,
                        "updated_at": new_form["updated_at"].isoformat() if new_form["updated_at"] else None
                    },
                    "trainer": {
                        "id": request_dict["trainer_id"],
                        "name": f"{request_dict['first_name']} {request_dict['last_name']}",
                        "email": request_dict["email"]
                    }
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to create form from request: {str(e)}")

# Form responses endpoints
@app.post("/forms/{form_id}/responses")
async def submit_form_response(form_id: int, response_data: dict, current_user: dict = Depends(get_current_user)):
    try:

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if form exists and is published
                cur.execute("""
                    SELECT id,
                           CASE WHEN status = 'published' THEN true ELSE COALESCE(is_published, false) END as is_published,
                           department_id
                    FROM forms WHERE id = %s
                """, (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                if not form.get("is_published", False):
                    raise HTTPException(status_code=404, detail="Form not published")

                # Check if user already responded (single-use forms)
                if current_user["id"]:
                    cur.execute("SELECT id FROM form_responses WHERE form_id = %s AND user_id = %s",
                               (form_id, current_user["id"]))
                    existing_response = cur.fetchone()
                    if existing_response:
                        raise HTTPException(status_code=400, detail="You have already responded to this form")

                # Insert the response
                response_json = json.dumps(response_data.get("responses", {}))

                cur.execute("""
                    INSERT INTO form_responses (form_id, user_id, response_data, is_anonymous, is_complete, submitted_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id, created_at, submitted_at
                """, (form_id, current_user["id"], response_json, False, True))

                response_record = cur.fetchone()
                conn.commit()

                # Check if this is a trainer evaluation and rating is below threshold
                # Get form details to check if it's evaluating a trainer
                cur.execute("""
                    SELECT f.title, f.trainer_id, u.first_name || ' ' || u.last_name as trainer_name,
                           u.email as trainer_email, u.supervisor_email, d.name as department_name
                    FROM forms f
                    JOIN users u ON f.trainer_id = u.id
                    LEFT JOIN departments d ON u.department_id = d.id
                    WHERE f.id = %s AND u.role = 'trainer' AND u.supervisor_email IS NOT NULL
                """, (form_id,))

                trainer_info = cur.fetchone()

                if trainer_info:
                    # Calculate average rating from the response
                    responses = response_data.get("responses", {})
                    ratings = []

                    for question_id, answer in responses.items():
                        if isinstance(answer, (int, float)) and 1 <= answer <= 5:
                            ratings.append(float(answer))
                        elif isinstance(answer, str) and answer.isdigit():
                            rating = float(answer)
                            if 1 <= rating <= 5:
                                ratings.append(rating)

                    if ratings:
                        avg_rating = sum(ratings) / len(ratings)

                        # Send supervisor notification if rating is below threshold
                        if avg_rating < LOW_RATING_THRESHOLD:
                            try:
                                send_supervisor_rating_alert(
                                    trainer_name=trainer_info["trainer_name"],
                                    trainer_email=trainer_info["trainer_email"],
                                    supervisor_email=trainer_info["supervisor_email"],
                                    current_rating=avg_rating,
                                    form_title=trainer_info["title"],
                                    department_name=trainer_info["department_name"] or "Unknown"
                                )
                            except Exception as email_error:
                                # Don't fail the response submission if email fails
                                pass

                return {
                    "success": True,
                    "message": "Form response submitted successfully",
                    "response": {
                        "id": response_record["id"],
                        "form_id": form_id,
                        "user_id": current_user["id"],
                        "submitted_at": (response_record.get("submitted_at") or response_record["created_at"]).isoformat(),
                        "user_email": current_user.get("email", "")
                    }
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to submit form response: {str(e)}")

# Check if user has completed a form
@app.get("/forms/{form_id}/completion-status")
async def check_form_completion(form_id: int, current_user: dict = Depends(get_current_user)):
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id FROM form_responses WHERE form_id = %s AND user_id = %s",
                           (form_id, current_user["id"]))
                response = cur.fetchone()

                return {
                    "success": True,
                    "is_completed": response is not None,
                    "form_id": form_id,
                    "user_id": current_user["id"]
                }
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to check form completion: {str(e)}")

@app.get("/forms/{form_id}/responses")
async def get_form_responses(form_id: int, current_user: dict = Depends(get_current_user)):
    try:

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                # Check if form exists and user has permission
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                # Check permissions
                # Admin can see all responses, trainers can see their own forms, trainees can see responses for forms in their department
                if current_user["role"] == "admin":
                    # Admin can see all
                    pass
                elif current_user["role"] == "trainer":
                    # Trainers can only see their own forms
                    if form["trainer_id"] != current_user["id"]:
                        raise HTTPException(status_code=403, detail="Access denied - trainers can only view their own form responses")
                elif current_user["role"] == "trainee":
                    # Trainees can see responses for forms in their department (for transparency)
                    if form["department_id"] != current_user["department_id"]:
                        raise HTTPException(status_code=403, detail="Access denied - you can only view responses for forms in your department")
                else:
                    raise HTTPException(status_code=403, detail="Access denied - insufficient permissions")

                # Get responses with user information
                cur.execute("""
                    SELECT fr.id, fr.form_id, fr.user_id, fr.response_data,
                           fr.is_anonymous, fr.is_complete, fr.created_at,
                           u.first_name || ' ' || u.last_name as user_name,
                           u.email as user_email,
                           d.name as department_name
                    FROM form_responses fr
                    JOIN users u ON fr.user_id = u.id
                    LEFT JOIN departments d ON u.department_id = d.id
                    WHERE fr.form_id = %s
                    ORDER BY fr.created_at DESC
                """, (form_id,))

                responses = []
                for row in cur.fetchall():
                    response_dict = dict(row)
                    # Parse response_data if it's a string
                    if isinstance(response_dict["response_data"], str):
                        try:
                            response_dict["response_data"] = json.loads(response_dict["response_data"])
                        except:
                            response_dict["response_data"] = {}

                    response_dict["created_at"] = response_dict["created_at"].isoformat() if response_dict["created_at"] else None
                    response_dict["submitted_at"] = response_dict["created_at"]
                    responses.append(response_dict)

                # Parse form_data if it exists
                form_data = form.get("form_data", {})
                if isinstance(form_data, str):
                    try:
                        form_data = json.loads(form_data)
                    except:
                        form_data = {}

                return {
                    "success": True,
                    "form": {
                        "id": form["id"],
                        "title": form["title"],
                        "description": form["description"],
                        "form_data": form_data
                    },
                    "responses": responses,
                    "total_responses": len(responses)
                }

    except HTTPException:
        raise
    except Exception as e:

        import traceback

        raise HTTPException(status_code=500, detail=f"Failed to get form responses: {str(e)}")

# Department endpoints
@app.get("/departments")
async def get_departments(current_user: dict = Depends(get_current_user)):
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM departments ORDER BY name")
                departments = []
                for row in cur.fetchall():
                    dept_dict = dict(row)
                    dept_dict["created_at"] = dept_dict["created_at"].isoformat() if dept_dict["created_at"] else None
                    departments.append(dept_dict)

                return {
                    "success": True,
                    "departments": departments
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get departments: {str(e)}")

# Demo users endpoint removed for production deployment

# User department management
@app.put("/users/departments")
async def update_user_departments(
    department_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user's departments (for trainers)"""
    try:

        department_ids = department_data.get('department_ids', [])
        user_id = current_user['id']

        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # For now, we'll update the user's primary department
                # In a full implementation, you might have a separate user_departments table
                if department_ids:
                    primary_dept = department_ids[0]  # Use first selected department as primary
                    cur.execute(
                        "UPDATE users SET department_id = %s, has_selected_departments = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (primary_dept, user_id)
                    )
                    conn.commit()

                return {
                    "success": True,
                    "message": "Departments updated successfully",
                    "department_ids": department_ids
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to update departments: {str(e)}")

# Report Generation Endpoints
@app.get("/reports/dashboard")
async def get_reports_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get comprehensive data for reports dashboard"""

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get comprehensive statistics
                cur.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM forms) as total_forms,
                        (SELECT COUNT(*) FROM forms WHERE status = 'published' OR is_published = true) as active_forms,
                        (SELECT COUNT(*) FROM form_responses) as total_responses,
                        (SELECT COUNT(*) FROM users WHERE role = 'trainer') as total_trainers,
                        (SELECT COUNT(*) FROM departments) as total_departments
                """)
                stats = cur.fetchone()

                # Get forms with response data for charts
                cur.execute("""
                    SELECT f.id, f.title, f.created_at, f.trainer_id, f.department_id,
                           u.name as trainer_name,
                           d.name as department_name,
                           COUNT(fr.id) as response_count
                    FROM forms f
                    LEFT JOIN users u ON f.trainer_id = u.id
                    LEFT JOIN departments d ON f.department_id = d.id
                    LEFT JOIN form_responses fr ON f.id = fr.form_id
                    WHERE f.status = 'published' OR f.is_published = true
                    GROUP BY f.id, f.title, f.created_at, f.trainer_id, f.department_id, u.name, d.name
                    ORDER BY f.created_at DESC
                    LIMIT 50
                """)
                forms_data = cur.fetchall()

                # Calculate response trends (last 7 days)
                response_trends = []
                for i in range(6, -1, -1):
                    date = datetime.now() - timedelta(days=i)
                    cur.execute("""
                        SELECT COUNT(*) as count FROM form_responses
                        WHERE DATE(created_at) = %s
                    """, (date.date(),))
                    count = cur.fetchone()['count']
                    response_trends.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'responses': count
                    })

                # Calculate average rating
                cur.execute(r"""
                    SELECT AVG(CAST(response_data->>'rating' AS FLOAT)) as avg_rating
                    FROM form_responses
                    WHERE response_data->>'rating' IS NOT NULL
                    AND response_data->>'rating' ~ '^[0-9]+\.?[0-9]*$'
                """)
                avg_rating_result = cur.fetchone()
                avg_rating = avg_rating_result['avg_rating'] if avg_rating_result and avg_rating_result['avg_rating'] else 0.0

                # Calculate response rate
                total_forms = stats['total_forms']
                total_responses = stats['total_responses']
                response_rate = round((total_responses / total_forms * 100), 1) if total_forms > 0 else 0

                return {
                    "success": True,
                    "stats": {
                        "totalForms": total_forms,
                        "activeForms": stats['active_forms'],
                        "totalResponses": total_responses,
                        "totalTrainers": stats['total_trainers'],
                        "totalDepartments": stats['total_departments'],
                        "averageRating": round(float(avg_rating), 1) if avg_rating else 0.0,
                        "responseRate": response_rate,
                        "responseTrends": response_trends,
                        "formsData": [dict(form) for form in forms_data]
                    }
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get reports data: {str(e)}")

@app.get("/reports/generate/{form_id}")
async def generate_form_report(form_id: int, current_user: dict = Depends(get_current_user)):
    """Generate comprehensive report for a specific form"""

    # Check permissions (admin or form creator)
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                form_dict = dict(form)

                # Check permissions
                if (current_user["role"] != "admin" and
                    form_dict["creator_id"] != current_user["id"] and
                    form_dict["trainer_id"] != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Generate report
                report = generate_comprehensive_report(form_id)

                if "error" in report:
                    raise HTTPException(status_code=500, detail=report["error"])

                return {
                    "success": True,
                    "report": report
                }

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.get("/reports/forms")
async def get_available_reports(current_user: dict = Depends(get_current_user)):
    """Get list of forms available for report generation"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if current_user["role"] == "admin":
                    # Admin sees all forms with responses
                    cur.execute("""
                        SELECT f.id, f.title, f.created_at,
                               COALESCE(f.status, CASE WHEN f.is_published = true THEN 'published' ELSE 'draft' END) as status,
                               f.due_date,
                               COUNT(fr.id) as response_count,
                               u.first_name || ' ' || u.last_name as creator_name,
                               d.name as department_name
                        FROM forms f
                        LEFT JOIN form_responses fr ON f.id = fr.form_id
                        LEFT JOIN users u ON f.creator_id = u.id
                        LEFT JOIN departments d ON f.department_id = d.id
                        WHERE (f.status = 'published' OR f.is_published = true)
                        GROUP BY f.id, f.title, f.created_at, f.status, f.is_published, f.due_date, u.first_name, u.last_name, d.name
                        ORDER BY f.created_at DESC
                    """)
                else:
                    # Trainers see only their forms
                    cur.execute("""
                        SELECT f.id, f.title, f.created_at,
                               COALESCE(f.status, CASE WHEN f.is_published = true THEN 'published' ELSE 'draft' END) as status,
                               f.due_date,
                               COUNT(fr.id) as response_count,
                               u.first_name || ' ' || u.last_name as creator_name,
                               d.name as department_name
                        FROM forms f
                        LEFT JOIN form_responses fr ON f.id = fr.form_id
                        LEFT JOIN users u ON f.creator_id = u.id
                        LEFT JOIN departments d ON f.department_id = d.id
                        WHERE (f.status = 'published' OR f.is_published = true) AND (f.creator_id = %s OR f.trainer_id = %s)
                        GROUP BY f.id, f.title, f.created_at, f.status, f.is_published, f.due_date, u.first_name, u.last_name, d.name
                        ORDER BY f.created_at DESC
                    """, (current_user["id"], current_user["id"]))

                forms = []
                for row in cur.fetchall():
                    form_dict = dict(row)
                    form_dict["created_at"] = form_dict["created_at"].isoformat() if form_dict["created_at"] else None
                    form_dict["due_date"] = form_dict["due_date"].isoformat() if form_dict.get("due_date") else None
                    forms.append(form_dict)

                return {
                    "success": True,
                    "forms": forms
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get available reports: {str(e)}")

@app.post("/reports/settings")
async def update_report_settings(settings_data: dict, current_user: dict = Depends(get_current_user)):
    """Update report generation settings (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # In a real implementation, you would store these in a database
        # For now, we'll just validate and return the settings

        valid_settings = {}

        if "threshold_low" in settings_data:
            threshold = float(settings_data["threshold_low"])
            if 1.0 <= threshold <= 5.0:
                valid_settings["threshold_low"] = threshold

        if "threshold_medium" in settings_data:
            threshold = float(settings_data["threshold_medium"])
            if 1.0 <= threshold <= 5.0:
                valid_settings["threshold_medium"] = threshold

        if "auto_generate" in settings_data:
            valid_settings["auto_generate"] = bool(settings_data["auto_generate"])

        if "email_alerts" in settings_data:
            valid_settings["email_alerts"] = bool(settings_data["email_alerts"])

        return {
            "success": True,
            "message": "Report settings updated successfully",
            "settings": valid_settings
        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@app.get("/reports/settings")
async def get_report_settings(current_user: dict = Depends(get_current_user)):
    """Get current report settings"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return {
        "success": True,
        "settings": {
            "threshold_low": REPORT_THRESHOLD_LOW,
            "threshold_medium": REPORT_THRESHOLD_MEDIUM,
            "auto_generate": REPORT_AUTO_GENERATE,
            "email_alerts": bool(SMTP_USERNAME and SMTP_PASSWORD)
        }
    }

def generate_department_report(department_id: int) -> dict:
    """Generate comprehensive department report with all forms, trainees, and trainer analysis"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get department details
                cur.execute("SELECT * FROM departments WHERE id = %s", (department_id,))
                department = cur.fetchone()
                if not department:
                    return {"error": "Department not found"}

                department_dict = dict(department)

                # Get all forms in this department
                cur.execute("""
                    SELECT f.*, u.first_name || ' ' || u.last_name as creator_name,
                           t.first_name || ' ' || t.last_name as trainer_name,
                           COUNT(fr.id) as response_count
                    FROM forms f
                    LEFT JOIN users u ON f.creator_id = u.id
                    LEFT JOIN users t ON f.trainer_id = t.id
                    LEFT JOIN form_responses fr ON f.id = fr.form_id
                    WHERE f.department_id = %s
                    GROUP BY f.id, u.first_name, u.last_name, t.first_name, t.last_name
                    ORDER BY f.created_at DESC
                """, (department_id,))

                forms = [dict(row) for row in cur.fetchall()]

                # Get all trainees in this department
                cur.execute("""
                    SELECT u.*, COUNT(fr.id) as total_responses,
                           AVG(CASE WHEN fr.response_data->>'overall_rating' ~ '^[0-9]+(\.[0-9]+)?$'
                               THEN CAST(fr.response_data->>'overall_rating' AS FLOAT) END) as avg_rating
                    FROM users u
                    LEFT JOIN form_responses fr ON u.id = fr.user_id
                    WHERE u.role = 'trainee' AND u.department_id = %s
                    GROUP BY u.id
                    ORDER BY u.first_name, u.last_name
                """, (department_id,))

                trainees = [dict(row) for row in cur.fetchall()]

                # Get all trainers in this department with performance metrics
                cur.execute("""
                    SELECT u.*,
                           COUNT(DISTINCT f.id) as forms_created,
                           COUNT(fr.id) as total_responses_received,
                           AVG(CASE WHEN fr.response_data->>'overall_rating' ~ '^[0-9]+(\.[0-9]+)?$'
                               THEN CAST(fr.response_data->>'overall_rating' AS FLOAT) END) as avg_rating,
                           COUNT(CASE WHEN fr.response_data->>'overall_rating' ~ '^[0-9]+(\.[0-9]+)?$'
                                 AND CAST(fr.response_data->>'overall_rating' AS FLOAT) >= 4.0 THEN 1 END) as high_ratings,
                           COUNT(CASE WHEN fr.response_data->>'overall_rating' ~ '^[0-9]+(\.[0-9]+)?$'
                                 AND CAST(fr.response_data->>'overall_rating' AS FLOAT) < 3.0 THEN 1 END) as low_ratings
                    FROM users u
                    LEFT JOIN forms f ON u.id = f.trainer_id AND f.department_id = %s
                    LEFT JOIN form_responses fr ON f.id = fr.form_id
                    WHERE u.role = 'trainer' AND u.department_id = %s
                    GROUP BY u.id
                    ORDER BY avg_rating DESC NULLS LAST, total_responses_received DESC
                """, (department_id, department_id))

                trainers = [dict(row) for row in cur.fetchall()]

                # Calculate department statistics
                total_forms = len(forms)
                total_responses = sum(form.get('response_count', 0) for form in forms)
                total_trainees = len(trainees)
                total_trainers = len([t for t in trainers if t.get('forms_created', 0) > 0])

                # Calculate average department rating
                all_ratings = []
                for form in forms:
                    if form['id']:
                        cur.execute("""
                            SELECT CAST(response_data->>'overall_rating' AS FLOAT) as rating
                            FROM form_responses
                            WHERE form_id = %s
                            AND response_data->>'overall_rating' ~ '^[0-9]+(\.[0-9]+)?$'
                        """, (form['id'],))
                        ratings = [row['rating'] for row in cur.fetchall()]
                        all_ratings.extend(ratings)

                avg_department_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0

                # Identify top and least performing trainers
                active_trainers = [t for t in trainers if t.get('total_responses_received', 0) > 0]
                top_trainer = max(active_trainers, key=lambda x: (x.get('avg_rating') or 0, x.get('total_responses_received', 0))) if active_trainers else None
                least_trainer = min(active_trainers, key=lambda x: (x.get('avg_rating') or 5, -x.get('total_responses_received', 0))) if active_trainers else None

                # Generate insights and recommendations
                insights = []
                recommendations = []

                if avg_department_rating < 3.0:
                    insights.append("Department performance is below average - immediate attention required")
                    recommendations.append("Conduct department-wide training review and improvement sessions")
                elif avg_department_rating < 3.5:
                    insights.append("Department performance needs improvement")
                    recommendations.append("Focus on trainer development and feedback quality")
                else:
                    insights.append("Department performance is satisfactory")

                if total_responses < total_forms * 5:  # Less than 5 responses per form on average
                    insights.append("Low response rate detected")
                    recommendations.append("Implement strategies to increase trainee engagement")

                if total_trainers > 0 and (total_responses / total_trainers) < 10:
                    insights.append("Low trainer utilization")
                    recommendations.append("Review trainer workload distribution")

                return {
                    "department": department_dict,
                    "statistics": {
                        "total_forms": total_forms,
                        "total_responses": total_responses,
                        "total_trainees": total_trainees,
                        "total_trainers": total_trainers,
                        "avg_department_rating": round(avg_department_rating, 2),
                        "response_rate": round((total_responses / (total_trainees * total_forms)) * 100, 1) if total_trainees > 0 and total_forms > 0 else 0
                    },
                    "forms": forms,
                    "trainees": trainees,
                    "trainers": trainers,
                    "top_performer": {
                        "trainer": top_trainer,
                        "metrics": {
                            "avg_rating": round(top_trainer.get('avg_rating', 0), 2) if top_trainer else 0,
                            "total_responses": top_trainer.get('total_responses_received', 0) if top_trainer else 0,
                            "forms_created": top_trainer.get('forms_created', 0) if top_trainer else 0
                        }
                    } if top_trainer else None,
                    "least_performer": {
                        "trainer": least_trainer,
                        "metrics": {
                            "avg_rating": round(least_trainer.get('avg_rating', 0), 2) if least_trainer else 0,
                            "total_responses": least_trainer.get('total_responses_received', 0) if least_trainer else 0,
                            "forms_created": least_trainer.get('forms_created', 0) if least_trainer else 0
                        }
                    } if least_trainer else None,
                    "insights": insights,
                    "recommendations": recommendations,
                    "generated_at": datetime.utcnow().isoformat(),
                    "report_version": "2.0"
                }

    except Exception as e:

        return {"error": str(e)}

@app.get("/reports/department/{department_id}")
async def generate_department_report_endpoint(department_id: int, current_user: dict = Depends(get_current_user)):
    """Generate comprehensive department report"""

    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        report = generate_department_report(department_id)

        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        return {
            "success": True,
            "report": report
        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to generate department report: {str(e)}")

@app.get("/reports/departments/list")
async def get_departments_for_reports(current_user: dict = Depends(get_current_user)):
    """Get list of departments available for report generation"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT d.*,
                           COUNT(DISTINCT f.id) as total_forms,
                           COUNT(DISTINCT u.id) as total_trainers,
                           COUNT(DISTINCT t.id) as total_trainees
                    FROM departments d
                    LEFT JOIN forms f ON d.id = f.department_id
                    LEFT JOIN users u ON d.id = u.department_id AND u.role = 'trainer'
                    LEFT JOIN users t ON d.id = t.department_id AND t.role = 'trainee'
                    GROUP BY d.id, d.name, d.code, d.description, d.created_at
                    ORDER BY d.name
                """)

                departments = []
                for row in cur.fetchall():
                    dept_dict = dict(row)
                    dept_dict["created_at"] = dept_dict["created_at"].isoformat() if dept_dict["created_at"] else None
                    departments.append(dept_dict)

                return {
                    "success": True,
                    "departments": departments
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get departments: {str(e)}")

@app.get("/reports/export/{form_id}")
async def export_report_pdf(form_id: int, current_user: dict = Depends(get_current_user)):
    """Export comprehensive report as PDF"""

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM forms WHERE id = %s", (form_id,))
                form = cur.fetchone()

                if not form:
                    raise HTTPException(status_code=404, detail="Form not found")

                form_dict = dict(form)

                # Check permissions
                if (current_user["role"] != "admin" and
                    form_dict.get("creator_id") != current_user["id"] and
                    form_dict.get("trainer_id") != current_user["id"]):
                    raise HTTPException(status_code=403, detail="Access denied")

                # Generate report data
                report = generate_comprehensive_report(form_id)

                if "error" in report:
                    raise HTTPException(status_code=500, detail=report["error"])

                # Generate PDF
                pdf_data = generate_pdf_report(report)

                # Create filename
                form_title = form_dict.get('title', 'Form Report').replace(' ', '_')
                filename = f"Report_{form_title}_{form_id}.pdf"

                return Response(
                    content=pdf_data,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )

    except HTTPException:
        raise
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to export PDF: {str(e)}")

@app.post("/admin/reset-system")
async def reset_system(current_user: dict = Depends(get_current_user)):
    """Reset the entire system - clear all data for fresh deployment (ADMIN ONLY)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Clear all data in correct order (respecting foreign key constraints)

                cur.execute("DELETE FROM form_responses")

                cur.execute("DELETE FROM forms")

                cur.execute("DELETE FROM form_requests")

                cur.execute("DELETE FROM users WHERE id != %s", (current_user["id"],))

                # Reset sequences

                cur.execute("ALTER SEQUENCE forms_id_seq RESTART WITH 1")
                cur.execute("ALTER SEQUENCE form_responses_id_seq RESTART WITH 1")
                cur.execute("ALTER SEQUENCE form_requests_id_seq RESTART WITH 1")
                cur.execute("ALTER SEQUENCE users_id_seq RESTART WITH 2")  # Keep admin as ID 1

                conn.commit()

                return {
                    "success": True,
                    "message": "System reset completed. All data cleared except admin user.",
                    "remaining_users": 1,
                    "admin_preserved": True
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to reset system: {str(e)}")

@app.get("/reports/settings")
async def get_report_settings(current_user: dict = Depends(get_current_user)):
    """Get report generation settings"""
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM reports_settings ORDER BY setting_name")
                settings = cur.fetchall()

                # Convert to dictionary format
                settings_dict = {}
                for setting in settings:
                    settings_dict[setting['setting_name']] = {
                        'value': setting['setting_value'],
                        'description': setting['description']
                    }

                return {
                    "success": True,
                    "settings": settings_dict
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to get report settings: {str(e)}")

@app.put("/admin/reports/settings")
async def update_report_settings(settings_data: dict, current_user: dict = Depends(get_current_user)):
    """Update report generation settings (Admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for setting_name, setting_value in settings_data.items():
                    cur.execute("""
                        INSERT INTO reports_settings (setting_name, setting_value, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (setting_name)
                        DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = CURRENT_TIMESTAMP
                    """, (setting_name, str(setting_value)))

                conn.commit()

                return {
                    "success": True,
                    "message": "Report settings updated successfully"
                }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to update report settings: {str(e)}")

# Start background scheduler when the app starts (disabled temporarily)
# start_background_scheduler()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("working_backend:app", host="127.0.0.1", port=8001, reload=True)
