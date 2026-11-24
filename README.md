# Healthcare Report Explainer

A deployable Python web app that extracts text from PDF and image-based medical
reports using local OCR, then displays lines explicitly marked high, low,
abnormal, or critical by the laboratory. It does not use a cloud AI service,
make diagnoses, or persist uploaded report contents.

## Run with Docker

```powershell
docker build -t healthcare-report-explainer .
docker run --rm -p 8000:8000 healthcare-report-explainer
```

Open [http://localhost:8000](http://localhost:8000). The readiness endpoint is
at `/health`; the interactive API documentation is at `/docs`.

After analyzing a report, use **Download PDF** to save a separate summary of
the lab-marked values. The PDF is generated in memory and is not retained by
the app.

## Run locally

Install Tesseract OCR and Python 3.12, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum upload size |
| `MAX_PDF_PAGES` | `20` | Maximum PDF page count |
| `OCR_LANGUAGE` | `eng` | Tesseract language code installed in the image |

The default container installs English Tesseract data. Configure ingress TLS,
authentication, rate limiting, and suitable privacy controls before exposing
the application publicly. Do not treat this educational tool as a substitute
for clinician review. OCR can misread values, and laboratory flags do not
establish a diagnosis. Multipart uploads may be temporarily buffered while a
request is handled, but the application does not retain them after the request.
