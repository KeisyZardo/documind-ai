
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz

app = FastAPI(
    title="DocuMind AI",
    description="API para análise inteligente de documentos PDF"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "DocuMind AI está funcionando!"
    }

@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo PDF."
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="O PDF deve ter no máximo 10 MB."
        )

    try:
        document = fitz.open(
            stream=content,
            filetype="pdf"
        )

        text = "\n".join(
            page.get_text() for page in document
        )

        document.close()

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Não foi possível encontrar texto no PDF."
            )

        return {
            "filename": file.filename,
            "pages": len(fitz.open(
                stream=content,
                filetype="pdf"
            )),
            "text": text
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler este PDF."
        )
      
