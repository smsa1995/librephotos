"""
Django settings for ownphotos project.

Generated by 'django-admin startproject' using Django 1.11.2.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import datetime
import os

for envvar in (
    "SECRET_KEY",
    "BACKEND_HOST",
    "DB_BACKEND",
    "DB_NAME",
    "DB_USER",
    "DB_PASS",
    "DB_HOST",
    "DB_PORT",
):
    if envvar not in os.environ:
        raise NameError(f"Environnement variable not set :{envvar}")


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ["SECRET_KEY"]
RQ_API_TOKEN = os.environ["SECRET_KEY"]
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DEBUG", "") == "1"

ALLOWED_HOSTS = ["backend", "localhost", os.environ.get("BACKEND_HOST")]

AUTH_USER_MODEL = "api.User"

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": datetime.timedelta(minutes=5),
    # 'ACCESS_TOKEN_LIFETIME': datetime.timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": datetime.timedelta(days=7),
}

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "api",
    "nextcloud",
    "rest_framework",
    "corsheaders",
    "chunked_upload",
    "django_extensions",
    "django_rq",
    "constance",
    "constance.backends.database",
]

CONSTANCE_BACKEND = "constance.backends.database.DatabaseBackend"
CONSTANCE_DATABASE_CACHE_BACKEND = "default"

# Must be less or egal of nb core CPU ( Nearly 2GB per process)
HEAVYWEIGHT_PROCESS_ENV = os.environ.get("HEAVYWEIGHT_PROCESS", "1")
HEAVYWEIGHT_PROCESS = (
    int(HEAVYWEIGHT_PROCESS_ENV) if HEAVYWEIGHT_PROCESS_ENV.isnumeric() else 1
)

CONSTANCE_CONFIG = {
    "ALLOW_REGISTRATION": (False, "Publicly allow user registration", bool),
    "ALLOW_UPLOAD": (
        os.environ.get("ALLOW_UPLOAD", "True")
        not in ("false", "False", "0", "f"),
        "Allow uploading files",
        bool,
    ),
    "SKIP_PATTERNS": (
        os.environ.get("SKIP_PATTERNS", ""),
        "Comma delimited list of patterns to ignore (e.g. '@eaDir,#recycle' for synology devices)",
        str,
    ),
    "HEAVYWEIGHT_PROCESS": (
        HEAVYWEIGHT_PROCESS,
        "Number of workers, when scanning pictures. This setting can dramatically affect the ram usage. Each worker needs 800MB of RAM. Change at your own will. Default is 1.",
        int,
    ),
    "MAP_API_KEY": (
        os.environ.get("MAPBOX_API_KEY", ""),
        "Map Box API Key",
        str,
    ),
    "IMAGE_DIRS": ("/data", "Image dirs list (serialized json)", str),
}


INTERNAL_IPS = ("127.0.0.1", "localhost", "192.168.1.100")

CORS_ALLOW_HEADERS = (
    "cache-control",
    "accept",
    "accept-encoding",
    "allow-credentials",
    "withcredentials",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
)

CORS_ALLOWED_ORIGINS = ("http://localhost:3000", "http://192.168.1.100:3000")

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_FILTER_BACKENDS": ("django_filters.rest_framework.DjangoFilterBackend",),
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 20000,
}

REST_FRAMEWORK_EXTENSIONS = {
    "DEFAULT_OBJECT_CACHE_KEY_FUNC": "rest_framework_extensions.utils.default_object_cache_key_func",
    "DEFAULT_LIST_CACHE_KEY_FUNC": "rest_framework_extensions.utils.default_list_cache_key_func",
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "api.middleware.FingerPrintMiddleware",
]

if DEBUG:
    MIDDLEWARE += ["silk.middleware.SilkyMiddleware"]
    INSTALLED_APPS += ["silk"]
    INSTALLED_APPS += ["drf_spectacular"]
    SPECTACULAR_SETTINGS = {
        "TITLE": "LibrePhotos",
        "DESCRIPTION": "Your project description",
        "VERSION": "1.0.0",
    }

ROOT_URLCONF = "ownphotos.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ownphotos.wsgi.application"

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends." + os.environ["DB_BACKEND"],
        "NAME": os.environ["DB_NAME"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASS"],
        "HOST": os.environ["DB_HOST"],
        "PORT": os.environ["DB_PORT"],
    },
}

if "REDIS_PATH" in os.environ:
    redis_path = "unix://" + os.environ["REDIS_PATH"]
    redis_path += "?db=" + os.environ.get("REDIS_DB", "0")
else:
    redis_path = "redis://" + os.environ["REDIS_HOST"]
    redis_path += ":" + os.environ["REDIS_PORT"] + "/1"

redis_password = os.environ["REDIS_PASS"] if "REDIS_PASS" in os.environ else ""
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": redis_path,
        "TIMEOUT": 60 * 60 * 24,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "PASSWORD": redis_password,
        },
    }
}

RQ_QUEUES = {
    "default": {
        "USE_REDIS_CACHE": "default",
        "DEFAULT_TIMEOUT": 60 * 60 * 24 * 7,
        "DB": 0,
    }
}

RQ = {
    "DEFAULT_RESULT_TTL": 60,
}

# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

# Allow to define data folder like /var/lib/librephotos

BASE_DATA = os.environ.get("BASE_DATA", "/")
BASE_LOGS = os.environ.get("BASE_LOGS", "/logs/")

STATIC_URL = "api/static/"
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DATA, "protected_media")
STATIC_ROOT = os.path.join(BASE_DIR, "static")
DATA_ROOT = os.path.join(BASE_DATA, "data")
IM2TXT_ROOT = os.path.join(BASE_DATA, "data_models", "im2txt")
PLACES365_ROOT = os.path.join(BASE_DATA, "data_models", "places365", "model")
CLIP_ROOT = os.path.join(BASE_DATA, "data_models", "clip-embeddings")
LOGS_ROOT = BASE_LOGS

CHUNKED_UPLOAD_PATH = ""
CHUNKED_UPLOAD_TO = os.path.join("chunked_uploads")

THUMBNAIL_SIZE_TINY = 100
THUMBNAIL_SIZE_SMALL = 200
THUMBNAIL_SIZE_MEDIUM = 400
THUMBNAIL_SIZE = 800
THUMBNAIL_SIZE_BIG = (2048, 2048)

FULLPHOTO_SIZE = (1000, 1000)

DEFAULT_FAVORITE_MIN_RATING = os.environ.get("DEFAULT_FAVORITE_MIN_RATING", 4)
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOW_CREDENTIALS = True

IMAGE_SIMILARITY_SERVER = "http://localhost:8002"


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
        },
    },
}
