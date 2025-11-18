import os
import json
import base64

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ==========================
# Configuração da Groq
# ==========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==========================
# FastAPI
# ==========================
app = FastAPI(title="Blaze IA – Double & Crash")

# CORS liberando tudo (p/ GitHub Pages, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnaliseResponse(BaseModel):
    acao: str
    confianca: float
    justificativa: str


@app.get("/")
def root():
    return {"status": "online", "msg": "Blaze IA backend OK"}


# ==========================
# Função que chama a Groq Vision
# ==========================
def chamar_groq_vision(image_bytes: bytes, modo: str) -> AnaliseResponse:
    """
    Envia o print para o modelo de visão da Groq e tenta extrair
    acao / confianca / justificativa em JSON.
    """

    if client is None:
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa="GROQ_API_KEY não configurada no servidor. IA desativada."
        )

    # Converte imagem para data URL base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    system_prompt = """
Você é uma IA especialista em padrões da Blaze (Double e Crash).
Analise SOMENTE o histórico visível no print.

OBJETIVO:
- MODO double: decidir se vale a pena buscar o BRANCO na próxima rodada.
- MODO crash: decidir se vale a pena entrar para buscar 2x (sair exatamente em 2x).

REGRAS IMPORTANTES:
- JAMAIS invente números. Use apenas o que está visível no print.
- Seja conservador: se não houver padrão forte, prefira NÃO OPERAR.
- Nunca prometa lucro garantido.

FORMATO DE RESPOSTA:
Responda estritamente em JSON, sem texto extra, neste formato:

{
  "acao": "BRANCO" ou "CRASH_2X" ou "NAO_OPERAR",
  "confianca": número entre 0 e 1,
  "justificativa": "texto curto explicando o motivo em português"
}
"""

    user_text = f"""
Modo atual: {modo}.

- Se o modo for "double", a ação deve ser:
  - "BRANCO"      -> se for uma boa oportunidade de buscar o branco
  - "NAO_OPERAR"  -> se não for um bom momento

- Se o modo for "crash", a ação deve ser:
  - "CRASH_2X"    -> se for uma boa oportunidade de buscar 2x
  - "NAO_OPERAR"  -> se não for um bom momento

Não use nenhum outro valor além destes.
Se estiver em dúvida, escolha "NAO_OPERAR".
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        # 👇 tipo correto pra visão na Groq
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )

        content = completion.choices[0].message.content.strip()
        data = json.loads(content)

        acao = str(data.get("acao", "NAO_OPERAR")).upper()
        confianca = float(data.get("confianca", 0.0))
        justificativa = str(data.get("justificativa", "")).strip()

        if acao not in {"BRANCO", "CRASH_2X", "NAO_OPERAR"}:
            acao = "NAO_OPERAR"

        return AnaliseResponse(
            acao=acao,
            confianca=max(0.0, min(1.0, confianca)),
            justificativa=justificativa
            or "IA não conseguiu explicar o motivo. Prefira não operar."
        )

    except Exception as e:
        return AnaliseResponse(
            acao="NAO_OPERAR",
            confianca=0.0,
            justificativa=f"Erro ao usar a IA: {e}. Prefira não operar nesta rodada."
        )


# ==========================
# Endpoint principal
# Aceita /api/analisar e /api/analisar/
# ==========================
@app.post("/api/analisar", response_model=AnaliseResponse)
@app.post("/api/analisar/", response_model=AnaliseResponse)
async def analisar_imagem(
    image: UploadFile = File(...),
    modo: str = Form("double"),
):
    """
    Recebe o print da Blaze + modo (double/crash) e retorna a decisão da IA.
    """
    image_bytes = await image.read()
    return chamar_groq_vision(image_bytes, modo)
