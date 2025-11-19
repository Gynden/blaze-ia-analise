# main.py
import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ==== CONFIG FASTAPI =====
app = FastAPI(title="Blaze IA – Double & Crash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== CLIENTE GROQ =====
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 🔴 MODEL ANTIGO (DEPRECADO)
# MODEL_ID = "llama-3.2-90b-vision-preview"

# ✅ MODEL NOVO (VISÃO)
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"


def encode_image_bytes(file_bytes: bytes) -> str:
    """Converte bytes da imagem em base64 (string)."""
    return base64.b64encode(file_bytes).decode("utf-8")


def montar_prompt(modo: str) -> str:
    """Texto base pro prompt, de acordo com o modo selecionado."""
    if modo == "crash":
        return (
            "Você é uma IA especialista em Blaze Crash. "
            "Seu foco é encontrar boas oportunidades de entrada para buscar apenas 2x, "
            "evitando entradas arriscadas.\n\n"
            "Analise o histórico/tela do print enviado e responda APENAS em JSON "
            "com as chaves: 'acao' (valores possíveis: 'CRASH_2X' ou 'NAO_OPERAR'), "
            "'confianca' (0 a 1) e 'justificativa' (texto curto em português explicando o motivo)."
        )
    else:
        # default = double
        return (
            "Você é uma IA especialista em Blaze Double focada em CAÇAR O BRANCO. "
            "Analise o histórico/tela do print enviado e diga se vale a pena "
            "entrar no branco na próxima rodada.\n\n"
            "Responda APENAS em JSON com as chaves: "
            "'acao' (valores possíveis: 'BRANCO' ou 'NAO_OPERAR'), "
            "'confianca' (0 a 1) e 'justificativa' (texto curto em português explicando o motivo)."
        )


@app.post("/api/analisar")
async def analisar_imagem(
    image: UploadFile = File(...),
    modo: str = Form("double"),  # "double" ou "crash"
):
    try:
        # Lê imagem enviada
        img_bytes = await image.read()
        base64_image = encode_image_bytes(img_bytes)

        # monta data URL pra visão
        mime = image.content_type or "image/png"
        data_url = f"data:{mime};base64,{base64_image}"

        prompt = montar_prompt(modo)

        system_message = {
            "role": "system",
            "content": (
                "Você SEMPRE deve responder em JSON puro, sem texto extra, "
                "seguindo exatamente as chaves pedidas no prompt do usuário."
            ),
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                },
            ],
        }

        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[system_message, user_message],
            temperature=0.3,
            max_completion_tokens=300,
            top_p=1,
            stream=False,
        )

        raw_content = completion.choices[0].message.content
        print("Resposta bruta da IA:", raw_content)

        # tenta fazer parse do JSON
        try:
            data = json.loads(raw_content)
        except Exception:
            data = {
                "acao": "NAO_OPERAR",
                "confianca": 0.0,
                "justificativa": (
                    "Não consegui interpretar a resposta da IA. "
                    "Prefira não entrar nesta rodada."
                ),
            }

        # garante campos básicos
        acao = data.get("acao", "NAO_OPERAR")
        confianca = float(data.get("confianca", 0) or 0)
        justificativa = data.get(
            "justificativa",
            "Análise indisponível. Prefira não entrar nesta rodada.",
        )

        return {
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
        }

    except Exception as e:
        return {
            "acao": "NAO_OPERAR",
            "confianca": 0.0,
            "justificativa": (
                f"Erro ao usar a IA: {type(e).__name__}('{e}'). "
                "Prefira não entrar nesta rodada."
            ),
        }


@app.get("/api/analisar")
async def status():
    return {
        "acao": "NAO_OPERAR",
        "confianca": 0.0,
        "justificativa": (
            "Backend Blaze IA está online. Envie um POST multipart/form-data "
            "para /api/analisar com o campo 'image' e, opcionalmente, 'modo' "
            "('double' ou 'crash')."
        ),
    }
