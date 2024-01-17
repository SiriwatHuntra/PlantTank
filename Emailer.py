import smtplib
from email.mime.text import MIMEText

def send_notification():
    # Set up your email details
    sender_email = "siriwat26room1@gmail.com"
    sender_password = "Shadow_2545"
    recipient_email = "siriwat26room1@gmail.com"

    # Compose the email message
    subject = "Plant Health Alert"
    body = "Your plant needs attention! It appears to be unhealthy."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())

