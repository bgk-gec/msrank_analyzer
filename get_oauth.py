import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

def authenticate():
    # 직접 파일 경로 설정
    CLIENT_SECRET_FILE = "C:\\Users\\kbg09\\.config\\client_secret.json"
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    TOKEN_FILE = "C:\\Users\\kbg09\\.config\\token.json"

    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        print("Loaded token from file.")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("Token refreshed.")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8000)
            print("New token obtained.")
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            print("Token saved.")
    
    return creds

if __name__ == "__main__":
    credentials = authenticate()
    print("Authentication successful!")
