
import os

import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

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
    return {"message": "DocuMind AI está funcionando!"}


class SummaryRequest(BaseModel):
    text: str
    filename: str = "Documento PDF"


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

        pages = len(document)
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
            "pages": pages,
            "text": text
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler este PDF."
        )


@app.post("/api/summarize")
async def summarize_document(request: SummaryRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Chave do Gemini não configurada."
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="O texto do documento está vazio."
        )

    if len(request.text) > 80000:
        raise HTTPException(
            status_code=413,
            detail="Documento muito extenso para esta versão."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Você é o assistente de análise de documentos "
                "do DocuMind AI. Faça um resumo claro e organizado "
                "em português do Brasil. Apresente o tema central, "
                "os pontos mais importantes e uma conclusão. "
                "Não invente informações que não estejam no texto.\n\n"
                f"Nome do documento: {request.filename}\n\n"
                f"Conteúdo:\n{request.text}"
            )
        )

        summary = response.text

        if not summary:
            raise ValueError("O Gemini não retornou um resumo.")

        return {
            "filename": request.filename,
            "summary": summary
        }

    except Exception as e:
        print("ERRO GEMINI:", repr(e))

        raise HTTPException(
            status_code=502,
            detail="Não foi possível gerar o resumo com o Gemini."
        )
