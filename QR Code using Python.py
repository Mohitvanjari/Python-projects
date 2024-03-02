import pyqrcode
from PIL import Image

link = "https://www.instagram.com/mohitvanjari?igsh=MXd0cmhoN3B2MW1vdg==/"
qr_code = pyqrcode.create(link)

# Save the QR code as a PNG file using the Pillow library
qr_code.png("instagram.png", scale=5)
