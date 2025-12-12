"""
Script to read a Google Docs document using the Google Docs API.
Make sure you have credentials.json set up in the same directory.
"""

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os.path
import pickle

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']

# The ID of the document.
DOCUMENT_ID = '1KG3n1oJiqpc60hEspGxQ904I1K5wahBEBihzfPW7LBw'

def read_structural_elements(elements):
    """Recursively extracts text from document elements."""
    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                if 'textRun' in elem:
                    text += elem.get('textRun').get('content')
        elif 'table' in value:
            table = value.get('table')
            for row in table.get('tableRows'):
                for cell in row.get('tableCells'):
                    text += read_structural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            toc = value.get('tableOfContents')
            text += read_structural_elements(toc.get('content'))
    return text

def main():
    """Shows basic usage of the Docs API.
    Prints the title and content of the document.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print('Error: credentials.json not found!')
                print('Please download credentials.json from Google Cloud Console.')
                return
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('docs', 'v1', credentials=creds)

    # Retrieve the documents contents from the Docs service.
    document = service.documents().get(documentId=DOCUMENT_ID).execute()

    print('=' * 80)
    print(f"Document Title: {document.get('title')}")
    print('=' * 80)
    print()
    
    content = document.get('body').get('content')
    text = read_structural_elements(content)
    print(text)
    
    # Save to a text file for reference
    with open('document_content.txt', 'w', encoding='utf-8') as f:
        f.write(f"Document Title: {document.get('title')}\n")
        f.write('=' * 80 + '\n\n')
        f.write(text)

if __name__ == '__main__':
    main()


