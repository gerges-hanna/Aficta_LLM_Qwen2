
📘 Documentation: Setting Up and Deploying a Flask Project on AWS EC2 (Aficta_LLM_Qwen2)

---

## 🧱 1. Preparing the Environment on EC2

### 🔧 Install Required Tools:
```bash
sudo apt-get update
sudo apt install python3 python3-pip nginx git -y
sudo apt install python3-venv
```
📌 Use these commands during the initial setup to install Python, Git, Nginx, and virtual environment support.

---

## 📂 Create Project Folder and Clone Code

```bash
mkdir flask-app
cd flask-app/
git clone https://github.com/gerges-hanna/Aficta_LLM_Qwen2.git
cd Aficta_LLM_Qwen2 
```

📌 Use these commands only once when deploying the project for the first time. Afterward, use `git pull` to update.

---

## 🐍 Create Virtual Environment and Run Manually

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 run.py
# CTRL+C to stop
```

📌 Useful for testing the app manually during development and verifying that everything works.

---

## 🚀 Run the App Using Gunicorn (Production Mode)

```bash
pip3 install gunicorn
gunicorn -w 3 -b 0.0.0.0:5000 run:app
# CTRL+C to stop
```

📌 Use Gunicorn for production-ready serving of your Flask app.

---

## 🌐 Configure Nginx to Forward Requests to Flask

### 1. Create the Nginx Config:
```bash
sudo mkdir /etc/conf.d/
sudo nano /etc/conf.d/flask-app.conf
```

📄 Example content of `flask-app.conf`:
```nginx
server {
    listen 80;
    server_name <ELASTIC_IP>;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 2. Link the config and start Nginx:
```bash
sudo ln /etc/conf.d/flask-app.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl restart nginx
sudo systemctl status nginx
```

📌 Always test Nginx config using `nginx -t` before restarting the service.

---

## ⚙️ Create systemd Service for Flask

```bash
sudo nano /etc/systemd/system/flask-app.service
```

📄 Example content of `flask-app.service`:
```ini
[Unit]
Description=Flask app
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/flask-app/Aficta_LLM_Qwen2
ExecStart=/home/ubuntu/flask-app/Aficta_LLM_Qwen2/venv/bin/gunicorn -w 3 -b 0.0.0.0:5000 run:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl restart flask-app
sudo systemctl status flask-app
```

📌 Use `daemon-reload` after any edits to the service file.

---

## 📦 After Pulling New Code (e.g., git pull)

```bash
sudo systemctl daemon-reload
sudo systemctl restart flask-app
curl http://127.0.0.1:5000/
```

📌 These steps ensure the app runs with the latest code.

---

## 📌 Notes:

- If your Elastic IP changes, update the following:
  - `server_name` inside `/etc/conf.d/flask-app.conf`
  - Re-link the config:
    ```bash
    sudo rm /etc/nginx/sites-enabled/flask-app.conf
    sudo ln /etc/conf.d/flask-app.conf /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl reload nginx
    ```

---

## ✅ Test the App:

```bash
curl http://127.0.0.1:5000/
```

Or open your browser: `http://<Elastic IP>`

---

## 📋 View Logs for Flask Service (It's helpful if project failed)

To check the most recent logs of your Flask application (managed via `systemd`), use the following command:

```bash
sudo journalctl -u flask-app --no-pager -n 50
```

This documentation is your reference for deploying, modifying, or troubleshooting your Flask app on AWS EC2.
