# 🎓 API & Authentication Study Guide
## From Zero to Production-Ready API

This guide explains everything about building the Anime Recommendation API from the ground up. No prior API experience required!

---

## 📚 Table of Contents

1. [What is an API?](#1-what-is-an-api)
2. [How HTTP Works](#2-how-http-works)
3. [FastAPI Basics](#3-fastapi-basics)
4. [Project Structure](#4-project-structure)
5. [Database Layer (SQLAlchemy)](#5-database-layer-sqlalchemy)
6. [Authentication Deep Dive](#6-authentication-deep-dive)
7. [Security Best Practices](#7-security-best-practices)
8. [Testing Your API](#8-testing-your-api)
9. [Industry Standards](#9-industry-standards)
10. [Code Walkthrough](#10-code-walkthrough)

---

## 1. What is an API?

### Definition
**API** (Application Programming Interface) is a way for different software programs to communicate with each other. Think of it like a waiter in a restaurant:

```
You (Client)  →  Waiter (API)  →  Kitchen (Server/Database)
    │                 │                    │
 "I want pizza"    Takes order      Makes pizza
    │                 │                    │
 Gets pizza    ←  Delivers it   ←   Sends it out
```

### REST API
Our API follows **REST** (Representational State Transfer) principles:

| Principle | Meaning | Example |
|-----------|---------|---------|
| **Stateless** | Server doesn't remember previous requests | Each request must include all needed info (like auth token) |
| **Client-Server** | Frontend and backend are separate | React app + Python API |
| **Uniform Interface** | Consistent URL structure | `/api/v1/users`, `/api/v1/anime` |
| **Resource-Based** | URLs represent things (nouns) | `/users/123` not `/getUser?id=123` |

### HTTP Methods
APIs use HTTP methods to indicate what action to perform:

| Method | Purpose | Example | Our Usage |
|--------|---------|---------|-----------|
| `GET` | Read data | Get user profile | `GET /api/v1/auth/me` |
| `POST` | Create data | Register new user | `POST /api/v1/auth/register` |
| `PUT` | Update (full replace) | Update entire profile | Not used yet |
| `PATCH` | Update (partial) | Change just password | Not used yet |
| `DELETE` | Remove data | Delete account | Not used yet |

---

## 2. How HTTP Works

### Request Structure
Every HTTP request has these parts:

```http
POST /api/v1/auth/login HTTP/1.1        ← Request Line (Method + Path + Version)
Host: localhost:8000                     ← Headers start here
Content-Type: application/x-www-form-urlencoded
Authorization: Bearer eyJhbGciOiJI...

username=test@example.com&password=secret  ← Body (the data)
```

### Response Structure
The server responds with:

```http
HTTP/1.1 200 OK                          ← Status Line
Content-Type: application/json           ← Headers
Set-Cookie: session=abc123

{"access_token": "eyJ...", "token_type": "bearer"}  ← Body (JSON data)
```

### Status Codes
Memorize these common ones:

| Code | Meaning | When Used |
|------|---------|-----------|
| `200` | OK | Request succeeded |
| `201` | Created | New resource created (registration) |
| `400` | Bad Request | Invalid input data |
| `401` | Unauthorized | Not logged in / bad token |
| `403` | Forbidden | Logged in but not allowed |
| `404` | Not Found | Resource doesn't exist |
| `422` | Unprocessable Entity | Validation failed |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server bug |

---

## 3. FastAPI Basics

### Why FastAPI?
FastAPI is a modern Python web framework with:
- ⚡ **Speed**: One of the fastest Python frameworks
- 📝 **Auto Documentation**: Generates Swagger UI automatically
- ✅ **Validation**: Automatic request/response validation
- 🔒 **Type Safety**: Uses Python type hints

### Basic Application Structure

```python
# main.py
from fastapi import FastAPI

app = FastAPI(
    title="Anime Recommendation API",  # Shows in docs
    version="1.0.0"
)

@app.get("/")  # Decorator defines the route
async def root():
    return {"message": "Hello World"}
```

### Key Concepts

#### 1. Path Parameters
Values in the URL path:
```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):  # FastAPI auto-converts to int
    return {"user_id": user_id}
```
URL: `GET /users/123` → `user_id = 123`

#### 2. Query Parameters
Values after `?` in URL:
```python
@app.get("/search")
async def search(query: str = "", limit: int = 20):
    return {"query": query, "limit": limit}
```
URL: `GET /search?query=naruto&limit=10`

#### 3. Request Body
JSON data sent with POST/PUT:
```python
from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str

@app.post("/register")
async def register(user: UserCreate):  # Auto-parsed from JSON body
    return {"email": user.email}
```

#### 4. Dependencies
Reusable logic injected into routes:
```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    # db is automatically provided and cleaned up
    return db.query(User).all()
```

---

## 4. Project Structure

Our API follows a clean, organized structure:

```
api/
├── main.py              # Application entry point, routes, middleware
├── config.py            # Configuration constants
├── database.py          # Database connection setup
├── models.py            # SQLAlchemy ORM models (database tables)
├── requirements.txt     # Python dependencies
│
├── auth/                # Authentication module
│   ├── __init__.py      # Makes it a Python package
│   ├── schemas.py       # Pydantic models (request/response shapes)
│   ├── security.py      # Password hashing, JWT functions
│   ├── dependencies.py  # get_current_user dependency
│   └── router.py        # Auth endpoints (/register, /login, /me)
│
├── inference/           # ML recommendation logic
│   ├── model.py         # RBM neural network
│   ├── recommender.py   # Get recommendations from model
│   └── ...
│
├── alembic/             # Database migrations
│   └── versions/        # Migration files
│
└── tests/               # Unit tests
    ├── test_api.py      # API endpoint tests
    └── test_auth.py     # Authentication tests
```

### Why This Structure?

| Directory | Purpose | Benefit |
|-----------|---------|---------|
| `auth/` | Group related code | Easy to find authentication code |
| `schemas.py` | Separate validation | Models ≠ API shapes |
| `dependencies.py` | Reusable auth | Don't repeat auth logic |
| `tests/` | Separate tests | Clear test organization |

---

## 5. Database Layer (SQLAlchemy)

### What is an ORM?
**ORM** (Object-Relational Mapping) lets you use Python classes instead of SQL:

```python
# Instead of SQL:
# SELECT * FROM users WHERE email = 'test@example.com';

# You write Python:
user = db.query(User).filter(User.email == "test@example.com").first()
```

### Our Database Setup

#### `database.py` - Connection Configuration
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Connection string format: dialect://user:pass@host:port/database
DATABASE_URL = os.getenv("DATABASE_URL")
# Example: "postgresql://anime_user:password@localhost:5432/anime_db"

# Base class that all models inherit from
Base = declarative_base()

# Engine manages the actual database connection
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # Check connection is alive
    pool_size=5,          # Keep 5 connections ready
    max_overflow=10       # Allow 10 more if needed
)

# SessionLocal creates new database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db      # Give session to the route
    finally:
        db.close()    # Always close when done
```

#### `models.py` - Database Tables as Classes
```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from api.database import Base

class User(Base):
    __tablename__ = "users"  # Actual table name in database
    
    # Each Column() becomes a database column
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship to other tables
    favorites = relationship("UserFavorite", back_populates="user", 
                           cascade="all, delete-orphan")
```

### Column Options Explained

| Option | Meaning | Example |
|--------|---------|---------|
| `primary_key=True` | Unique identifier | `id` column |
| `unique=True` | No duplicate values | `email` column |
| `index=True` | Faster searches | Frequently searched columns |
| `nullable=False` | Required field | `email` cannot be NULL |
| `default=value` | Auto-fill if not provided | `is_active=True` |
| `ForeignKey("table.column")` | Links to another table | `user_id` → `users.id` |

### Database Migrations (Alembic)
Migrations track changes to your database schema:

```bash
# Create a new migration after changing models.py
alembic revision --autogenerate -m "Add avatar column"

# Apply all pending migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

---

## 6. Authentication Deep Dive

### The Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REGISTRATION                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client                           Server                             │
│    │                                 │                               │
│    │  POST /register                 │                               │
│    │  {email, password, username}    │                               │
│    │ ─────────────────────────────►  │                               │
│    │                                 │  1. Validate input            │
│    │                                 │  2. Check email not taken     │
│    │                                 │  3. Hash password             │
│    │                                 │  4. Save to database          │
│    │                                 │                               │
│    │  201 Created                    │                               │
│    │  {id, email, username}          │                               │
│    │ ◄─────────────────────────────  │                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            LOGIN                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client                           Server                             │
│    │                                 │                               │
│    │  POST /login                    │                               │
│    │  username=email&password=pass   │                               │
│    │ ─────────────────────────────►  │                               │
│    │                                 │  1. Find user by email        │
│    │                                 │  2. Verify password hash      │
│    │                                 │  3. Generate JWT token        │
│    │                                 │                               │
│    │  200 OK                         │                               │
│    │  {access_token, token_type}     │                               │
│    │ ◄─────────────────────────────  │                               │
│    │                                 │                               │
│    │  Client stores token locally    │                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      PROTECTED REQUEST                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client                           Server                             │
│    │                                 │                               │
│    │  GET /me                        │                               │
│    │  Authorization: Bearer <token>  │                               │
│    │ ─────────────────────────────►  │                               │
│    │                                 │  1. Extract token from header │
│    │                                 │  2. Decode & verify JWT       │
│    │                                 │  3. Get user_id from token    │
│    │                                 │  4. Fetch user from database  │
│    │                                 │                               │
│    │  200 OK                         │                               │
│    │  {id, email, username}          │                               │
│    │ ◄─────────────────────────────  │                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Password Security

#### Why Hash Passwords?
**NEVER** store plain passwords. If database is breached, hackers get all passwords.

```python
# ❌ BAD - Plain password storage
user.password = "secret123"  # If leaked, attacker knows password

# ✅ GOOD - Hashed password storage
user.hashed_password = "$2b$12$LQv3c..."  # One-way hash, can't reverse
```

#### How Bcrypt Works
```python
import bcrypt

def get_password_hash(password: str) -> str:
    """
    bcrypt.gensalt() generates a random "salt" (random data).
    This means same password → different hash each time.
    Protects against rainbow table attacks.
    """
    salt = bcrypt.gensalt()  # e.g., b'$2b$12$LQv3c1yqBWVHxkd0...'
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    bcrypt stores the salt IN the hash, so it can verify.
    Hash format: $2b$12$[22 char salt][31 char hash]
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )
```

**Example:**
```python
>>> get_password_hash("secret123")
'$2b$12$LQv3c1yqBWVHxkd0LH/PHeQJvPxL.pF0aE7rEv7rq2eL3NeHhEWHK'
#    │    │                  │
#    │    │                  └── The actual hash
#    │    └── Salt (random, unique per password)
#    └── Cost factor (2^12 iterations, makes brute-force slow)

>>> verify_password("secret123", "$2b$12$LQv3c1yq...")
True
>>> verify_password("wrongpassword", "$2b$12$LQv3c1yq...")
False
```

### JWT (JSON Web Tokens)

#### What is a JWT?
A JWT is a self-contained token that proves identity. It has 3 parts:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzA5MTIzNDU2fQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
│                                      │                                                                                        │
└──────── Header ──────────────────────┴───────────────────────── Payload ──────────────────────────────────────────────────────┴── Signature ──
```

**Decoded:**
```json
// Header (algorithm info)
{
  "alg": "HS256",
  "typ": "JWT"
}

// Payload (the actual data - called "claims")
{
  "sub": "1",              // Subject (user ID)
  "email": "test@example.com",
  "exp": 1709123456        // Expiration timestamp
}

// Signature (prevents tampering)
// HMACSHA256(base64(header) + "." + base64(payload), SECRET_KEY)
```

#### How JWT Security Works

```python
# Creating a token (server-side)
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    
    # Sign with secret key - only server knows this key
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

# Verifying a token (server-side)
def decode_access_token(token: str) -> dict | None:
    try:
        # This verifies the signature using SECRET_KEY
        # If token was tampered with, signature won't match
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        return None  # Invalid or expired token
```

**Why JWTs are Secure:**
1. **Signature**: If anyone modifies the payload, the signature becomes invalid
2. **Expiration**: Tokens expire, limiting damage if stolen
3. **Stateless**: Server doesn't need to store sessions

**Why `sub` Must Be a String:**
The JWT standard (RFC 7519) specifies that the `sub` (subject) claim should be a string. The `python-jose` library enforces this:

```python
# ❌ This will fail when decoding
create_access_token(data={"sub": 1})  # Integer

# ✅ This works correctly
create_access_token(data={"sub": str(user.id)})  # String
```

### OAuth2 with Password Flow

FastAPI uses OAuth2 "password flow" for simple username/password auth:

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# This tells FastAPI where to get tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
    #           │
    #           └── Expects: username=...&password=... (form data, not JSON)
):
    # form_data.username contains the email (OAuth2 standard uses "username")
    # form_data.password contains the password
    ...
```

**Why form data instead of JSON?**
OAuth2 specification requires `application/x-www-form-urlencoded` format for the token endpoint. This is an industry standard.

---

## 7. Security Best Practices

### Our Implementation Follows These:

#### ✅ 1. Environment Variables for Secrets
```python
# ❌ BAD - Hardcoded secret
SECRET_KEY = "mysecretkey123"

# ✅ GOOD - From environment
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-key")
```

#### ✅ 2. Password Hashing with Bcrypt
- Uses slow algorithm (intentionally)
- Includes salt (random data)
- Cost factor makes brute-force impractical

#### ✅ 3. JWT with Expiration
```python
expire = datetime.now(timezone.utc) + timedelta(minutes=30)
```
Short expiration limits damage if token is stolen.

#### ✅ 4. Rate Limiting
```python
@limiter.limit("30/minute")  # Max 30 requests per minute
async def recommend(request: Request, ...):
    ...
```
Prevents brute-force attacks and abuse.

#### ✅ 5. Input Validation
```python
class UserCreate(BaseModel):
    email: EmailStr  # Must be valid email format
    password: str = Field(..., min_length=8)  # At least 8 chars
```

#### ✅ 6. SQL Injection Prevention
SQLAlchemy ORM automatically parameterizes queries:
```python
# ✅ Safe - SQLAlchemy handles escaping
db.query(User).filter(User.email == user_input).first()

# ❌ Dangerous - Raw SQL with string formatting
db.execute(f"SELECT * FROM users WHERE email = '{user_input}'")
```

#### ✅ 7. CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Only allow specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)
```

#### ✅ 8. Error Messages Don't Leak Info
```python
# ✅ GOOD - Generic message
raise HTTPException(status_code=401, detail="Incorrect email or password")

# ❌ BAD - Reveals which field is wrong
raise HTTPException(status_code=401, detail="Password incorrect")
# This tells attacker the email exists!
```

### What We Could Add:

| Feature | Purpose | Priority |
|---------|---------|----------|
| Refresh Tokens | Allow token renewal without re-login | High |
| Password Reset | Email-based password recovery | High |
| Account Lockout | Lock after N failed attempts | Medium |
| HTTPS Only | Encrypt all traffic | Critical for production |
| Audit Logging | Track security events | Medium |
| 2FA | Two-factor authentication | Optional |

---

## 8. Testing Your API

### Why Testing Matters
- **Catch bugs early** before users see them
- **Prevent regressions** when changing code
- **Documentation** - tests show how to use the API
- **Confidence** to refactor and improve

### Test Structure

```python
# test_auth.py

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Setup: Create test database, mock ML models
    with TestClient(app) as client:
        yield client
    # Teardown: Clean up database

class TestAuthRegister:
    """Group related tests together."""
    
    def test_register_success(self, client):
        """Test successful registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "new@example.com",
                "password": "securepassword123"
            }
        )
        assert response.status_code == 201
        assert response.json()["email"] == "new@example.com"
    
    def test_register_duplicate_email(self, client, existing_user):
        """Test that duplicate emails are rejected."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": existing_user.email,  # Already exists
                "password": "anypassword123"
            }
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
```

### Test Types

| Type | Purpose | Example |
|------|---------|---------|
| **Unit Test** | Test single function | `test_get_password_hash()` |
| **Integration Test** | Test components together | `test_login_endpoint()` |
| **End-to-End Test** | Test full user flow | Register → Login → Use API |

### Running Tests

```bash
# Run all tests
pytest api/tests/ -v

# Run specific test file
pytest api/tests/test_auth.py -v

# Run specific test
pytest api/tests/test_auth.py::TestAuthLogin::test_login_success -v

# Run with coverage report
pytest --cov=api --cov-report=html
```

### Our Test Coverage
```
api/tests/test_api.py   - 10 tests (search, recommend, health)
api/tests/test_auth.py  - 10 tests (register, login, me endpoint)
Total: 20 tests ✓
```

---

## 9. Industry Standards

### API Versioning
```
/api/v1/users    ← Version 1
/api/v2/users    ← Version 2 (can coexist)
```

Why? Allows breaking changes without affecting existing clients.

### RESTful URL Design

```
# Resources (nouns, not verbs)
GET    /api/v1/users          # List users
POST   /api/v1/users          # Create user
GET    /api/v1/users/123      # Get user 123
PUT    /api/v1/users/123      # Update user 123
DELETE /api/v1/users/123      # Delete user 123

# Sub-resources
GET    /api/v1/users/123/favorites    # User 123's favorites

# Filtering with query parameters
GET    /api/v1/anime?genre=action&limit=20&offset=0
```

### Response Format Consistency

```json
// Success response
{
  "data": { ... },
  "meta": {
    "total": 100,
    "page": 1,
    "per_page": 20
  }
}

// Error response
{
  "detail": "Email already registered",
  "code": "DUPLICATE_EMAIL"
}
```

### HTTP Status Code Usage
- `2xx` - Success
- `4xx` - Client error (bad request, unauthorized)
- `5xx` - Server error (bugs, database down)

### Documentation
FastAPI auto-generates:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## 10. Code Walkthrough

### File-by-File Explanation

#### `api/main.py` - The Heart of the Application

```python
# ============ IMPORTS ============
from fastapi import FastAPI, HTTPException, Request, APIRouter
from slowapi import Limiter  # Rate limiting

# ============ CONFIGURATION ============
limiter = Limiter(key_func=get_remote_address)  # Rate limit by IP
v1_router = APIRouter(prefix="/api/v1")  # Version prefix

# ============ PYDANTIC MODELS ============
class RecommendRequest(BaseModel):
    liked_anime: list[str]  # Input validation

# ============ APPLICATION LIFECYCLE ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup/shutdown."""
    # Startup: Load ML model, data
    yield
    # Shutdown: Cleanup resources

# ============ CREATE APP ============
app = FastAPI(title="Anime Recommendation API", lifespan=lifespan)

# ============ MIDDLEWARE ============
app.add_middleware(CORSMiddleware, ...)  # Cross-origin requests

@app.middleware("http")
async def log_requests(request, call_next):
    """Log every request."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} | {response.status_code}")
    return response

# ============ ROUTES ============
@v1_router.post("/recommend")
@limiter.limit("30/minute")
async def recommend(request: Request, body: RecommendRequest):
    ...

# ============ ROUTER REGISTRATION ============
v1_router.include_router(auth_router)  # Add auth routes
app.include_router(v1_router)  # Add all v1 routes to app
```

#### `api/auth/schemas.py` - Data Shapes

```python
from pydantic import BaseModel, EmailStr, Field, ConfigDict

class UserCreate(BaseModel):
    """What client sends to register."""
    email: EmailStr  # Validates email format
    password: str = Field(..., min_length=8)  # At least 8 chars
    username: str | None = Field(None, min_length=3, max_length=50)

class UserResponse(BaseModel):
    """What server returns (no password!)."""
    id: int
    email: str
    username: str | None
    is_active: bool
    
    # Allows creating from SQLAlchemy model
    model_config = ConfigDict(from_attributes=True)
```

#### `api/auth/security.py` - Security Functions

```python
import bcrypt
from jose import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def get_password_hash(password: str) -> str:
    """One-way hash - can't reverse."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(plain: str, hashed: str) -> bool:
    """Check if password matches hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict) -> str:
    """Create signed JWT."""
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(minutes=30)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    """Verify and decode JWT."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
```

#### `api/auth/dependencies.py` - Reusable Auth Logic

```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),  # Extract from header
    db: Session = Depends(get_db)
) -> User:
    """
    This dependency:
    1. Gets token from "Authorization: Bearer <token>" header
    2. Decodes and validates the JWT
    3. Fetches user from database
    4. Returns user or raises 401
    """
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user
```

#### `api/auth/router.py` - Auth Endpoints

```python
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check email not taken
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(400, "Email already registered")
    
    # Create user with hashed password
    new_user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )
    db.add(new_user)
    db.commit()
    return new_user

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), ...):
    user = db.query(User).filter(User.email == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")
    
    token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer")

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Protected route - requires valid token."""
    return current_user
```

---

## 🎯 Quick Reference

### Making API Requests

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secretpass123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=user@example.com&password=secretpass123"

# Use token
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJI..."

# Search anime
curl "http://localhost:8000/api/v1/search-anime?query=naruto&limit=10"

# Get recommendations
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_anime": ["Naruto", "One Piece"]}'
```

### Common Debugging

| Problem | Solution |
|---------|----------|
| 401 Unauthorized | Check token is valid, not expired |
| 422 Validation Error | Check request body matches schema |
| 500 Server Error | Check server logs for traceback |
| CORS Error | Check `ALLOWED_ORIGINS` includes your frontend URL |

---

## 📖 Further Learning

1. **FastAPI Documentation**: https://fastapi.tiangolo.com/
2. **SQLAlchemy Tutorial**: https://docs.sqlalchemy.org/en/20/tutorial/
3. **JWT Introduction**: https://jwt.io/introduction
4. **OAuth2 Simplified**: https://aaronparecki.com/oauth-2-simplified/
5. **REST API Best Practices**: https://restfulapi.net/

---

*This guide was created for the Anime Recommendation API project. All 20 tests pass! ✅*
