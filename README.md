# 📄 PDF Compression

Welcome to **PDF Compression**, a lightweight and efficient tool to reduce the size of PDF files for easier storage and sharing. This project demonstrates PDF compression using **Python**, **Flask**, and libraries like **pypdf** and **Pillow**.

---

## 🚀 Features

- 📉 Compress PDFs by deflating streams, merging duplicate objects and re-encoding images (Low / Medium / High image quality)
- 🔍 Side-by-side comparison of strategies; every output is re-opened and checked before it is offered
- ⚡ Fast and efficient file processing
- 📂 Upload and download compressed PDFs
- 🖥️ Clean and responsive web interface using Bootstrap
- 🛡️ Server-side validation: PDF header, 50 MB limit, encrypted files rejected; results are deleted after 1 hour

---

## 🛠️ Tech Stack

| Technology | Description                          |
| ---------- | ------------------------------------ |
| Python     | Core backend logic                   |
| Flask      | Lightweight web framework            |
| pypdf      | PDF manipulation                     |
| Pillow     | Image compression inside PDFs        |
| Bootstrap  | Frontend framework for responsive UI |
| HTML/CSS   | Structure and styling                |
| JavaScript | Basic interactivity                  |

---

## 📸 Screenshots

<div align="center">
    <img width="70%" src="Media/ss1.png" alt="Screen Shot">
</div>

---

## ⚙️ Installation

1️⃣ Clone this repository:

```bash
git clone https://github.com/Divyansh3105/PDF-Compression.git
```

2️⃣ Navigate into the project folder:

```bash
cd PDF-Compression
```

3️⃣ Install the required packages (a virtualenv is recommended):

```bash
pip install -r requirements.txt
```

4️⃣ Run the Flask app:

```bash
python compress.py
```

---

## 🌐 Usage

- Open your browser and go to `http://127.0.0.1:5000/`
- Upload a PDF, select a compression level, and download the compressed file.
- Set `FLASK_DEBUG=1` to enable debug mode (off by default; never expose it publicly).
- Run tests with `pip install pytest && pytest`.

---

## 🤝 Contribution

Pull requests are welcome! Feel free to fork the repository and submit improvements.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🔗 Links

- [🔗 GitHub Repo](https://github.com/Divyansh3105/PDF-Compression)

## Contact

- 📧 Email: divyanshgarg3105@gmail.com
- 🌐 GitHub: [@Divyansh3105](https://github.com/Divyansh3105)
