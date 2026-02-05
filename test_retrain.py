
import os
import urllib.request
import urllib.parse
import json

# Ensure uploads dir exists
os.makedirs('uploads', exist_ok=True)

# Create CSV
csv_path = 'uploads/test_students.csv'
with open(csv_path, 'w') as f:
    f.write('Name,Study Hours,Sleep Hours,Attendance,Social Media,Exercise,Stress\n')
    f.write('TestUser_Alpha,8,8,95,1,5,3\n')
    f.write('TestUser_Beta,2,5,60,6,1,8\n')
    f.write('TestUser_Gamma,5,7,85,3,3,5\n')

print(f"Created {csv_path}")

# Trigger Retrain Endpoint
url = 'http://localhost:5000/api/admin/retrain-models'
print(f"Calling {url}...")

try:
    req = urllib.request.Request(url, method='POST')
    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
        print(f"Response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
