import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="Blaze IA – Double & Crash")

# CORS liberado pro seu front (GitHub Pages, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente da Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Modelo NOVO de visão da Groq
MODEL_NAME = "llama-3.2-90b-vision-preview"


@app.get("/")
async def root():
    return {"status": "ok", "msg": "Blaze IA API online"}


async def file_to_base64(upload: UploadFile) -> str:
    """
    Converte o arquivo enviado em base64 (string) para mandar pra IA.
    """
    content = await upload.read()
    return base64.b64encode(content).decode("utf-8")


@app.post("/api/analisar")
async def analisar(
    image: UploadFile = File(...),
    modo: str = Form("double"),  # "double" ou "crash"
):
    """
    Recebe um print da Blaze + modo de jogo e retorna ação da IA:
    - Double: focar em BRANCO
    - Crash: focar em 2x
    """

    # garante valor conhecido
    modo = modo.lower().strip()
    if modo not in ("double", "crash"):
        modo = "double"

    # descrição amigável pro prompt
    if modo == "double":
        modo_descricao = (
            "Blaze Double, focando apenas em decidir se vale a pena entrar "
            "buscando o BRANCO (white)."
        )
    else:
        modo_descricao = (
            "Blaze Crash, focando apenas em decidir se vale a pena entrar "
            "buscando multiplicador de 2x."
        )

    try:
        img_b64 = await file_to_base64(image)

        prompt_text = f"""
Você é uma IA que analisa prints da Blaze no modo {modo_descricao}.

REGRAS IMPORTANTES:
- Analise somente o histórico visível no print.
- Considere sequência de cores/valores, padrões recentes e risco.
- Se o cenário estiver muito ruim ou incerto, recomende NÃO ENTRAR.

Sua resposta DEVE ser SOMENTE um JSON válido, sem texto antes ou depois, no seguinte formato:

{{
  "acao": "BRANCO" | "CRASH_2X" | "NAO_OPERAR",
  "confianca": 0.0 a 1.0,
  "justificativa": "texto curto em português explicando o motivo"
}}

- Para Blaze Double (modo="double"), use:
  - "BRANCO"    -> quando enxergar oportunidade de entrar no branco
  - "NAO_OPERAR" -> quando não for seguro entrar

- Para Blaze Crash (modo="crash"), use:
  - "CRASH_2X"  -> quando enxergar boa chance de bater 2x
  - "NAO_OPERAR" -> quando não for seguro entrar

Se não conseguir analisar por qualquer motivo, responda:
{{
  "acao": "NAO_OPERAR",
  "confianca": 0.0,
  "justificativa": "Não consegui analisar o print com segurança. Prefira não operar nesta rodada."
}}
        """.strip()

        # chamada ao modelo de visão
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Você é uma IA especialista em Blaze que responde sempre em JSON válido.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=400,
        )

        raw_content = completion.choices[0].message.content
        # tenta interpretar como JSON
        try:
            data = json.loads(raw_content)
        except Exception:
            # fallback se a IA mandar qualquer coisa fora do padrão
            data = {
                "acao": "NAO_OPERAR",
                "confianca": 0.0,
                "justificativa": (
                    "Erro ao interpretar a resposta da IA. "
                    "Prefira não operar nesta rodada."
                ),
            }

        # saneia campos básicos pra evitar erro no front
        acao = str(data.get("acao", "NAO_OPERAR")).upper()
        try:
            confianca = float(data.get("confianca", 0.0))
        except Exception:
            confianca = 0.0
        justificativa = str(
            data.get(
                "justificativa",
                "Mercado indefinido. Prefira não operar nesta rodada.",
            )
        )

        # clampa confiança entre 0 e 1
        if confianca < 0:
            confianca = 0.0
        if confianca > 1:
            confianca = 1.0

        return {
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
            "modo": modo,
        }

    except Exception as e:
        # log básico no backend e resposta segura
        print("Erro ao usar IA:", repr(e))
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": (
                f"Erro ao usar a IA. Detalhe: {repr(e)}. "
                "Prefira não operar nesta rodada."
            ),
            "modo": modo,
        }
