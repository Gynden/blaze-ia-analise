import os
import base64
import json
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq

# ---------------------------------------------------------
# Configuração básica
# ---------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Variável de ambiente GROQ_API_KEY não encontrada.")

client = Groq(api_key=GROQ_API_KEY)

# modelo visão atual da Groq
MODEL_NAME = "llama-3.2-11b-vision-preview"

app = FastAPI(title="Blaze IA - Double & Crash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se quiser, depois limita pro domínio do GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Rotas simples de teste
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "online", "message": "Blaze IA backend OK"}


# ---------------------------------------------------------
# Função utilitária: converte imagem pra data URL (base64)
# ---------------------------------------------------------

async def file_to_data_url(file: UploadFile) -> str:
    content = await file.read()
    mime_type = file.content_type or "image/png"
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------
# Rota principal: /api/analisar
# ---------------------------------------------------------

@app.post("/api/analisar")
async def analisar_imagem(
    image: UploadFile = File(...),
    modo: str = Form("double")  # "double" ou "crash" vindo do front
):
    """
    Recebe um print da Blaze + modo de jogo e retorna:
    {
      "acao": "ENTRAR_BRANCO" | "ENTRAR_2X" | "NAO_ENTRAR",
      "confianca": float 0..1,
      "justificativa": "texto curto"
    }
    """
    try:
        image_url = await file_to_data_url(image)

        # Prompt para a IA
        system_prompt = (
            "Você é uma IA especialista em Blaze Double (foco no BRANCO) "
            "e Blaze Crash (foco em buscar 2x). "
            "Você SEMPRE deve responder em JSON válido, sem texto extra, "
            "no formato:\n"
            "{"
            '"acao": "ENTRAR_BRANCO" ou "ENTRAR_2X" ou "NAO_ENTRAR", '
            '"confianca": número entre 0 e 1, '
            '"justificativa": "texto curto em português explicando o motivo"'
            "}\n"
            "Não use acentos estranhos, não use markdown, não coloque ```."
        )

        user_text = (
            f"Modo de jogo atual: {modo}.\n"
            "- Se o modo for 'double', analise o histórico e o padrão de cores e "
            "diga se vale a pena tentar ENTRAR_BRANCO ou se é melhor NAO_ENTRAR.\n"
            "- Se o modo for 'crash', analise o gráfico de multiplicadores e diga "
            "se vale a pena tentar ENTRAR_2X (buscar 2x) ou se é melhor NAO_ENTRAR.\n"
            "Considere sempre gestão de risco e evite entradas arriscadas demais."
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},  # força JSON válido
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
            temperature=0.15,
            max_tokens=200,
        )

        raw_content = completion.choices[0].message.content

        # Agora deve ser JSON puro por causa do response_format
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            # fallback se mesmo assim vier zoado
            return {
                "acao": "NAO_ENTRAR",
                "confianca": 0.0,
                "justificativa": (
                    "Não consegui interpretar a resposta da IA. "
                    "Prefira não entrar nesta rodada."
                ),
            }

        acao = data.get("acao", "NAO_ENTRAR")
        confianca = data.get("confianca", 0.0)
        justificativa = data.get("justificativa") or (
            "IA não enviou justificativa. Prefira não entrar nesta rodada."
        )

        # garante tipo numérico e faixa 0..1
        try:
            confianca = float(confianca)
        except (ValueError, TypeError):
            confianca = 0.0

        if confianca < 0:
            confianca = 0.0
        if confianca > 1:
            confianca = 1.0

        return {
            "acao": acao,
            "confianca": confianca,
            "justificativa": justificativa,
        }

    except Exception as e:
        # Aqui qualquer erro de Groq, rede, etc.
        return {
            "acao": "NAO_ENTRAR",
            "confianca": 0.0,
            "justificativa": (
                f"Erro ao usar a IA: {str(e)}. "
                "Prefira não entrar nesta rodada."
            ),
        }
