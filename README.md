# Facial-Attendece-using-Siamese-Network

This project explain how to create a sianese network for facial attendence in a web page using flask.
We are using app.py to create a web page and html files(index.html, register.html) for the display of the webpage.
It doesnt need much training since we are using siamese model.
Using 5-10 pictures per person is enough but i used one 3 per head and got the output,
I wont guarantte that we will per perfect result for 3 pictures every time better to use a minimum of 5 per head.

★ I changed all of my actual pictures to famous figures in indian politics
## 💻 Features

- Siamese model for face verification
- Live webcam face capture
- Attendance logging in CSV
- Local Flask-based web interface
- By default your system’s can_mark_attendance function only allows one mark per 24 hours. So if John’s face is recognized once today, further appearances of John’s face will return “Attendance already marked for --- today.”

## 🚀 How to Run

1. Clone this repo:
```bash
git clone https://github.com/YOUR_USERNAME/siamese-attendance.git
cd siamese-attendance
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Start the flask app:
```bash
python app.py
```
or
```bash
python3 app.py
```
5. Open in browser
```bash
http://localhost:5000
```
