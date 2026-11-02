
import os
import asyncio

import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from groq import AsyncGroq


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


def is_temporary_error(error):
    error_text = str(error).upper()

    temporary_codes = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "CONNECTION"
    ]

    return any(code in error_text for code in temporary_codes)


@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
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

    except Exception as e:
        print("ERRO PDF:", repr(e))

        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler este PDF."
        )


@app.post("/api/summarize")
async def summarize_document(request: SummaryRequest):

    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
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

    prompt = (
        "Você é o assistente de análise de documentos do DocuMind AI. "
        "Produza um resumo objetivo, equilibrado e fácil de entender.\n\n"

        "ESTRUTURA OBRIGATÓRIA:\n"
        "1. TEMA CENTRAL: explique o assunto em até 2 frases.\n"
        "2. PONTOS PRINCIPAIS: apresente de 3 a 5 tópicos "
        "com as informações essenciais.\n"
        "3. CONCLUSÃO: finalize em até 2 frases.\n\n"

        "REGRAS:\n"
        "- Escreva aproximadamente 100 a 150 palavras no total.\n"
        "- Seja direto e evite explicações longas.\n"
        "- Elimine repetições e detalhes secundários.\n"
        "- Preserve os fatos mais importantes.\n"
        "- Não invente informações.\n"
        "- Não reproduza CPF, e-mail, telefone ou endereço, "
        "a menos que sejam indispensáveis para compreender o documento.\n"
        "- Não revele dados pessoais desnecessariamente.\n"
        "- Responda em português do Brasil.\n\n"

        f"Documento: {request.filename}\n\n"
        f"Conteúdo:\n{request.text}"
    )

    try:
        # PRIMEIRO PROVEDOR: GEMINI

        client = genai.Client(api_key=gemini_key)

        for attempt in range(3):

            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config={
                        "max_output_tokens": 500
                    }
                )

                summary = response.text

                if not summary:
                    raise ValueError(
                        "O Gemini não retornou um resumo."
                    )

                print("Resumo gerado pelo Gemini.")

                return {
                    "filename": request.filename,
                    "summary": summary,
                    "provider": "gemini"
                }

            except Exception as e:

                if not is_temporary_error(e):
                    raise

                print(
                    f"Gemini indisponível. "
                    f"Tentativa {attempt + 1} de 3."
                )

                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))

                else:
                    print(
                        "Limite de tentativas do Gemini atingido. "
                        "Acionando provedor reserva."
                    )

        # SEGUNDO PROVEDOR: GROQ

        groq_key = os.getenv("GROQ_API_KEY")

        if not groq_key:
            raise RuntimeError(
                "Gemini indisponível e chave do Groq não configurada."
            )

        groq_client = AsyncGroq(api_key=groq_key)

        groq_response = await groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=500
        )

        summary = groq_response.choices[0].message.content

        if not summary:
            raise ValueError(
                "O Groq não retornou um resumo."
            )

        print("Resumo gerado pelo Groq.")

        return {
            "filename": request.filename,
            "summary": summary,
            "provider": "groq"
        }

    except Exception as e:

        print("ERRO NA GERAÇÃO DO RESUMO:", repr(e))

        raise HTTPException(
            status_code=502,
            detail=(
                "Não foi possível gerar o resumo neste momento. "
                "Tente novamente em instantes."
            )
        )
