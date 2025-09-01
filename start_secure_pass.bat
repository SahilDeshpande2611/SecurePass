@echo off

start cmd /k "cd C:\Users\PRATIK\SECURE_PASS\SECURE_PASS_BACKEND && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "cd C:\Users\PRATIK\SECURE_PASS\SECURE_PASS_FRONTEND && npm run dev"
