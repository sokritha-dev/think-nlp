# app/messages/upload_messages.py

# ✅ Positive
UPLOAD_SUCCESS = "File uploaded successfully."
DUPLICATE_FILE_FOUND = "This file has been uploaded before. Returning existing file."

# ❌ Errors
INVALID_CSV_FORMAT = "Invalid CSV format. Please upload a properly formatted CSV."
MISSING_REVIEW_COLUMN = "Missing required 'review' column in the uploaded file."
UPLOAD_FAILED = "Failed to upload the file due to internal server error."
